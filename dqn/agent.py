import random
import numpy as np
import torch
import torch.nn.functional as F

from dqn.network import QNetwork
from dqn.replay_buffer import ReplayBuffer
from dqn.hyperparam import Hyperparameters


class Agent:

	def select_action(self, state, q, eps, num_actions):
		"""eps-greedy: with probability eps take a random action (explore),
		otherwise take the action with the highest predicted Q-value (exploit)."""
		if random.random() < eps:
			return random.randrange(num_actions)
		with torch.no_grad():
			s = torch.as_tensor(np.array(state, dtype=np.float32)).unsqueeze(0)
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

		# === TD target ===
		# Use the frozen target net to value s_next. No gradient flows through
		# here because we are using the target as a fixed answer key, not as
		# something to train on.
		with torch.no_grad():
			q_next_all = q_target(s_next) # (B, |A|)
			a_next = q_next_all.argmax(dim=1, keepdim=True) # (B, 1)
			q_next = q_next_all.gather(1, a_next).squeeze(1) # (B,)

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
		# max_grad_norm clipping would go here in production.
		optimizer.step()

		return float(loss.item())
