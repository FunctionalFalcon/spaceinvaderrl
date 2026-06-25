import random
import numpy as np
import torch
import torch.nn.functional as F

from scratch.network import QNetwork
from scratch.replay_buffer import ReplayBuffer
from scratch.hyperparam import Hyperparameters


class Agent:

	def select_action(self, state, q, eps, num_actions):
		"""eps-greedy: with probability eps take a random action (explore),
		otherwise take the action with the highest predicted Q-value (exploit).

		Device handling: we read the device off the network's first parameter
		so the input tensor lives on the same device as `q`'s weights. Without
		this, training on CUDA raises:
		 RuntimeError: Input type (torch.FloatTensor) and weight type
		 (torch.cuda.FloatTensor) should be the same
		because np.array() returns a CPU tensor and q.to('cuda') lives on GPU.
		"""
		if random.random() < eps:
			return random.randrange(num_actions)
		# next(...) on parameters() yields the first Parameter; .device gives
		# e.g. device(type='cuda', index=0) or device(type='cpu').
		q_device = next(q.parameters()).device
		with torch.no_grad():
			s = torch.as_tensor(np.array(state, dtype=np.float32),
								device=q_device).unsqueeze(0)
		return int(q(s).argmax(dim=1).item())

	def epsilon_at(self, t, hp):
		"""Linear decay from eps_start to eps_end over the first eps_frac
		fraction of training. Stays pinned at eps_end after the decay window."""
		decay_steps = max(1, int(hp.eps_frac * hp.total_steps))
		progress = max(0.0, 1.0 - t / decay_steps)
		return hp.eps_end + (hp.eps_start - hp.eps_end) * progress

	def train_step(
		self,
		q_online,
		q_target, # frozen copy of q_online, refreshed every target_update steps
		optimizer,
		buffer,
		hp,
	):
		# Sample a minibatch of (s, a, r, s_next, done) from the replay buffer.
		s, a, r, s_next, done = buffer.sample(hp.batch_size)
		# ReplayBuffer.sample() returns CPU tensors (it never knows what device
		# the network lives on). Move the batch to q_online's device so the
		# forward pass and loss compute on the same hardware. Without this,
		# CUDA training raises "Input type (torch.FloatTensor) and weight type
		# (torch.cuda.FloatTensor) should be the same" the first time we try
		# to learn (after learning_starts steps).
		q_device = next(q_online.parameters()).device
		s = s.to(q_device)
		a = a.to(q_device)
		r = r.to(q_device)
		s_next = s_next.to(q_device)
		done = done.to(q_device)

		# === TD target (Double DQN) ===
		# Standard DQN uses the target net for both action selection and
		# action valuation, which causes a systematic overestimation bias
		# (same optimistic net "picks" and "values" -> bias compounds).
		# Double DQN (van Hasselt et al. 2015) decouples the two: the online
		# net picks the action, the target net values it. This reduces the
		# maximization bias and empirically yields ~30-50% higher mean
		# reward on Atari at the same training budget.
		with torch.no_grad():
			# ONLINE picks: which action looks best in s_next?
			a_next = q_online(s_next).argmax(dim=1, keepdim=True) # (B, 1)
			# TARGET values: how good is that action under the frozen net?
			q_next = q_target(s_next).gather(1, a_next).squeeze(1) # (B,)

		# Bellman target: r + gamma * Q(s_next, a*) when not done, else just r.
		# (1 - done) zeros out the bootstrapped future term on terminal steps.
		target = r + hp.gamma * q_next * (1.0 - done)

		# === Prediction ===
		# What did the online net think Q(s, a) was? (gradients DO flow here.)
		q_pred = q_online(s).gather(1, a.unsqueeze(1)).squeeze(1) # (B,)

		# === Huber loss ===
		# Squared for small errors, linear for big ones. Keeps one outlier
		# reward spike from dominating the gradient.
		loss = F.smooth_l1_loss(q_pred, target)

		# === Optimize ===
		optimizer.zero_grad()
		loss.backward()
		# Gradient clipping: cap the L2 norm of the gradient at hp.max_grad_norm
		# (default 10). One outlier reward spike (e.g. clearing an entire wave)
		# used to produce a huge gradient that could blow up the network weights
		# and require a full restart. Clipping scales the gradient down so the
		# worst-case update magnitude is bounded. Mnih 2015 used this exact
		# value (10.0); it's the standard DQN recipe.
		torch.nn.utils.clip_grad_norm_(q_online.parameters(), hp.max_grad_norm)
		optimizer.step()

		# Q-value diagnostics. Detach so we don't grow the autograd graph
		# just for logging. These are the canonical DQN health signals:
		# if mean_q blows up, the network is diverging; if it stays near 0
		# forever, the network isn't learning anything useful.
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
		}
