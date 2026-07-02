import random
import numpy as np
import torch
import torch.nn.functional as F

from scratch.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from scratch.hyperparam import Hyperparameters


class Agent:
    """Agent for Dueling DQN with Epsilon-greedy exploration and Prioritized Replay."""

    def select_action(
        self,
        state: np.ndarray,
        q,
        eps: float,
        num_actions: int,
    ) -> int:
        """
        Epsilon-greedy action selection.

        Exploration (random action) is cleanly separated from exploitation
        (greedy on Q-values). This is the standard Mnih 2015 approach and
        avoids the Noisy Nets / PER oscillation problem.
        """
        if random.random() < eps:
            return random.randrange(num_actions)
        q_device = next(q.parameters()).device
        with torch.no_grad():
            s = torch.as_tensor(np.array(state, dtype=np.float32), device=q_device).unsqueeze(0)
        return int(q(s).argmax(dim=1).item())

    def select_action_greedy(
        self,
        state: np.ndarray,
        q,
    ) -> int:
        """
        Greedy (pure exploitation) action selection. Used during evaluation.
        """
        q_device = next(q.parameters()).device
        with torch.no_grad():
            s = torch.as_tensor(np.array(state, dtype=np.float32), device=q_device).unsqueeze(0)
        return int(q(s).argmax(dim=1).item())

    def epsilon_at(self, t: int, hp: Hyperparameters) -> float:
        """Linear decay from eps_start to eps_end over eps_frac fraction of training."""
        decay_steps = max(1, int(hp.eps_frac * hp.total_steps))
        progress = max(0.0, 1.0 - t / decay_steps)
        return hp.eps_end + (hp.eps_start - hp.eps_end) * progress

    def train_step(
        self,
        q_online,
        q_target,
        optimizer,
        buffer,
        hp: Hyperparameters,
        step: int = 0,
    ):
        """
        Train one step on a batch from the replay buffer.

        Args:
            q_online:  the network being updated
            q_target:  frozen target network
            optimizer:  Adam optimizer
            buffer:    PrioritizedReplayBuffer or ReplayBuffer
            hp:        hyperparameters
            step:      current training step (for beta scheduling)

        Returns:
            dict with loss, mean_q, max_q, min_q, td_errors (for priority update)
        """
        is_prioritized = isinstance(buffer, PrioritizedReplayBuffer)

        if is_prioritized:
            s, a, r, s_next, done, is_weights, tree_indices = buffer.sample(hp.batch_size)
            buffer.update_beta_on_step(step)
        else:
            s, a, r, s_next, done = buffer.sample(hp.batch_size)
            is_weights = None
            tree_indices = None

        q_device = next(q_online.parameters()).device
        s = s.to(q_device)
        a = a.to(q_device)
        r = r.to(q_device)
        s_next = s_next.to(q_device)
        done = done.to(q_device)
        if is_weights is not None:
            is_weights = is_weights.to(q_device)

        # === TD target (Double DQN) ===
        # Online net picks the action, target net values it.
        with torch.no_grad():
            a_next = q_online(s_next).argmax(dim=1, keepdim=True)
            q_next = q_target(s_next).gather(1, a_next).squeeze(1)

        target = r + hp.gamma * q_next * (1.0 - done)

        # === Prediction ===
        q_pred = q_online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # === Loss with optional importance sampling weights ===
        td_errors = (target - q_pred).detach().cpu().numpy()
        if is_weights is not None:
            # Weighted Huber loss: apply IS weights element-wise
            # Note: F.smooth_l1_loss returns per-element loss, shape (B,)
            per_element_loss = F.smooth_l1_loss(q_pred, target, reduction='none')
            loss = (per_element_loss * is_weights).mean()
        else:
            loss = F.smooth_l1_loss(q_pred, target)

        # === Optimize ===
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_online.parameters(), hp.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            q_for_log = q_pred.detach()
            mean_q = float(q_for_log.mean().item())
            max_q = float(q_for_log.max().item())
            min_q = float(q_for_log.min().item())

        return {
            "loss": float(loss.item()),
            "mean_q": mean_q,
            "max_q": max_q,
            "min_q": min_q,
            "td_errors": td_errors,
            "tree_indices": tree_indices,
            "is_prioritized": is_prioritized,
        }

    def post_train_step(self, buffer, metrics):
        """
        Call after each train_step when using PrioritizedReplayBuffer.
        Updates priorities based on observed TD errors.
        """
        if metrics.get("is_prioritized") and metrics.get("td_errors") is not None:
            buffer.update_priorities(
                metrics["tree_indices"],
                metrics["td_errors"],
            )