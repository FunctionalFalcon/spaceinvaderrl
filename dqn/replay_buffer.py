import numpy as np
import torch

"""
We need a replay buffer.
If we train by order, consecutive gradient steps see almost-indentical inputs
That could bias the weights toward whatever it just saw.
Bsically we make random minibatches to break temporal correlation, so each gradient step is independent.
"""

class ReplayBuffer:
  """ring buffer
  (obs, action, reward, next_obs, done) tuples.
  bellman
  """

  def __init__(self, capacity: int, obs_shape: tuple):
    self.capacity = capacity
    self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
    self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
    self.actions = np.zeros(capacity, dtype = np.int64)
    self.rewards = np.zeros(capacity, dtype = np.float32)
    self.dones = np.zeros(capacity, dtype = np.float32)
    self.idx = 0  # next write position
    self.size = 0 # number of valid entries
    # oldest experience gets overwritten in place


  def push(self, s, a, r, s_next, done):
    self.obs[self.idx] = s
    self.next_obs[self.idx] = s_next
    self.actions[self.idx] = a
    self.rewards[self.idx] = r
    self.dones[self.idx] = float(done)
    self.idx = (self.idx + 1) % self.capacity
    self.size = min(self.size +1, self.capacity)
    # s (b, 4, 84, 84) float32
    # a (b,) int64
    # r (b,) float32
    # s_next (b, 4, 84, 84) float32
    # done (b, ) float32

  def sample(self, batch_size: int):
    i = np.random.randint(0, self.size, size = batch_size)
    return (
      torch.as_tensor(self.obs[i]),
      torch.as_tensor(self.actions[i]),
      torch.as_tensor(self.rewards[i]),
      torch.as_tensor(self.next_obs[i]),
      torch.as_tensor(self.dones[i]),
    )

"""
b = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))
 check shape

for _ in range(50): b.push(np.zeros((4,84,84)), 1, 0.0, np.zeros((4,84,84)), False)
o, a, r, o2, d = b.sample(4)
assert o.shape == (4, 4, 84, 84)
assert a.dtype == torch.int64 or torch.long
assert d.dtype == torch.float32
"""