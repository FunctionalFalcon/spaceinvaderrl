import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# SumTree — binary tree for O(log N) prioritized sampling
# ─────────────────────────────────────────────────────────────────────────────
# Each leaf stores one transition's priority. Internal nodes store the sum
# of their children's priorities.
#
# Operations:
#   add(priority)  → returns leaf index of new entry     O(log N)
#   sample(p)      → returns leaf index where p lands    O(log N)
#   update(idx, p) → update a leaf's priority            O(log N)
#
# Total storage: 2 * capacity - 1 nodes.

class SumTree:
    """Binary tree for prioritized sampling. All operations are O(log N)."""

    def __init__(self, capacity: int):
        # Cap at next power of 2 for clean tree layout
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        # Map leaf index → tree node index
        self._leaf_offset = capacity
        self._size = 0  # must be initialized before add() is called

    def _total(self) -> float:
        """Root holds the sum of all priorities."""
        return self.tree[0]

    def add(self, priority: float) -> int:
        """Add a new entry with given priority. Returns the leaf index (0..capacity-1)."""
        idx = self._leaf_offset + self.size
        self.tree[idx] = priority
        self._propagate(idx)
        self.size = min(self.size + 1, self.capacity)
        return idx - self._leaf_offset

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample `batch_size` indices proportional to their priority.

        Returns:
            indices: leaf indices (0..capacity-1), length batch_size
            priorities: the priority value at each sampled index, length batch_size
            tree_indices: internal tree node indices, length batch_size
        """
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float32)
        tree_indices = np.empty(batch_size, dtype=np.int64)

        # Segment the [0, total] range into batch_size equal chunks
        # p = segment_start + uniform(0, segment_size)
        total = self._total()
        if total == 0:
            # All priorities are zero — fall back to uniform sampling
            uniform_indices = np.random.randint(0, self.size, size=batch_size)
            for i, u_idx in enumerate(uniform_indices):
                tree_idx = self._leaf_offset + u_idx
                indices[i] = u_idx
                priorities[i] = self.tree[tree_idx]
                tree_indices[i] = tree_idx
            return indices, priorities, tree_indices

        segment_size = total / batch_size
        for i in range(batch_size):
            p = segment_size * i + np.random.uniform(0, segment_size)
            tree_idx = self._retrieve(0, p)
            leaf_idx = tree_idx - self._leaf_offset
            indices[i] = leaf_idx
            priorities[i] = self.tree[tree_idx]
            tree_indices[i] = tree_idx
        return indices, priorities, tree_indices

    def update(self, tree_indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update multiple leaf priorities. O(k log N) for k updates."""
        for ti, p in zip(tree_indices, priorities):
            if p <= 0:
                p = 1e-6  # clamp to avoid zero-probability
            self.tree[ti] = p
            self._propagate(ti)

    def __repr__(self):
        return f"SumTree(capacity={self.capacity}, total={self._total():.4f}, size={self.size})"

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _propagate(self, idx: int) -> None:
        """Propagate priority change from leaf up to root."""
        while idx > 0:
            idx = (idx - 1) // 2
            left = 2 * idx + 1
            right = 2 * idx + 2
            self.tree[idx] = self.tree[left] + self.tree[right]

    def _retrieve(self, idx: int, p: float) -> int:
        """Walk down the tree to find the leaf containing position p."""
        left = 2 * idx + 1
        right = 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if self.tree[left] + self.tree[right] == 0:
            return left  # degenerate case
        if p < self.tree[left]:
            return self._retrieve(left, p)
        else:
            return self._retrieve(right, p - self.tree[left])

    # Add size tracking
    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, val: int) -> None:
        self._size = val


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────
# Samples transitions with probability proportional to their TD error magnitude.
# Uses a SumTree for O(log N) operations.
#
# Key features:
#   - Initial priorities = max_priority (1.0) for all new transitions
#   - IS (importance sampling) weights to correct sampling bias
#   - Priority = |TD_error|^alpha, where alpha=0.6 by default
#
# Reference: "Prioritized Experience Replay" (Schaul et al., 2015)

class PrioritizedReplayBuffer:
    """
    Ring buffer with prioritized sampling.

    Stores (s, a, r, s_next, done) tuples.
    Sampling is weighted by |TD_error|^alpha.
    Returns importance-sampling weights for unbiased gradient updates.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_end: float = 1.0,
        beta_frac: float = 0.5,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            capacity: max transitions stored
            obs_shape: shape of observation, e.g. (4, 84, 84)
            alpha: prioritization exponent — 0 = uniform, 1 = pure TD-error weighting
            beta: initial IS correction exponent — 0 = no correction, 1 = full correction
            beta_end: final beta after beta_frac of training steps
            epsilon: small constant added to priorities (avoids zero-probability)
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.alpha = alpha
        self.beta = beta
        self.beta_end = beta_end
        self.beta_frac = beta_frac
        self.epsilon = epsilon
        self._step = 0  # training step counter (call `update_beta_on_step`)

        # Pre-allocate storage
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._idx = 0      # next write position (ring buffer)
        self._size = 0     # valid entries

        # SumTree for prioritized sampling
        self.tree = SumTree(capacity)
        # max_priority starts at 1.0; grows as we observe larger TD errors
        self.max_priority = 1.0

    @property
    def size(self) -> int:
        return self._size

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s_next: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition. Uses max_priority for new entries."""
        self.obs[self._idx] = s
        self.next_obs[self._idx] = s_next
        self.actions[self._idx] = a
        self.rewards[self._idx] = r
        self.dones[self._idx] = float(done)

        # New transitions get max_priority so they're replayed early
        self.tree.add(self.max_priority)

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a prioritized batch.

        Returns:
            (s, a, r, s_next, done, is_weights, tree_indices)
            is_weights: normalized importance-sampling weights for the loss
            tree_indices: internal SumTree indices (needed for priority update)
        """
        if self._size < batch_size:
            batch_size = self._size

        # Sample from SumTree
        indices, priorities, tree_indices = self.tree.sample(batch_size)

        # Compute importance-sampling weights: w_i = (P(i) * N) ^ (-beta)
        # Normalized by max weight to prevent large gradients
        beta = self._current_beta()
        probs = priorities / (self.tree._total() + 1e-10)
        is_weights = (self._size * probs + 1e-10) ** (-beta)
        is_weights = is_weights / (is_weights.max() + 1e-10)  # normalize
        is_weights = is_weights.astype(np.float32)

        return (
            torch.as_tensor(self.obs[indices]),
            torch.as_tensor(self.actions[indices]),
            torch.as_tensor(self.rewards[indices]),
            torch.as_tensor(self.next_obs[indices]),
            torch.as_tensor(self.dones[indices]),
            torch.as_tensor(is_weights),
            indices.astype(np.int64),
        )

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """
        Update priorities after a training step.

        Args:
            tree_indices: internal tree indices returned by sample()
            td_errors: absolute TD error for each sampled transition
        """
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.tree.update(tree_indices, priorities)
        self.max_priority = max(self.max_priority, priorities.max())

    def update_beta_on_step(self, step: int) -> None:
        """Update beta (IS exponent) as training progresses. Call each step."""
        self._step = step

    def _current_beta(self) -> float:
        """Linear schedule from beta to beta_end over first beta_frac of steps."""
        frac = min(1.0, self._step / max(1, int(self.beta_frac * 10_000_000)))
        return self.beta + (self.beta_end - self.beta) * frac

    def __repr__(self) -> str:
        return (
            f"PrioritizedReplayBuffer(size={self._size}/{self.capacity}, "
            f"alpha={self.alpha}, beta={self._current_beta():.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Original uniform ReplayBuffer (kept for comparison)
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Ring buffer with uniform random sampling."""

    def __init__(self, capacity: int, obs_shape: tuple):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def push(self, s, a, r, s_next, done):
        self.obs[self._idx] = s
        self.next_obs[self._idx] = s_next
        self.actions[self._idx] = a
        self.rewards[self._idx] = r
        self.dones[self._idx] = float(done)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        i = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.as_tensor(self.obs[i]),
            torch.as_tensor(self.actions[i]),
            torch.as_tensor(self.rewards[i]),
            torch.as_tensor(self.next_obs[i]),
            torch.as_tensor(self.dones[i]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────
"""
# Test SumTree
tree = SumTree(8)
for p in [1, 2, 3, 4]:
    tree.add(p)
assert tree._total() == 10.0
idx, pri, _ = tree.sample(4)
assert len(idx) == 4

# Test uniform fallback
tree2 = SumTree(4)
tree2.add(0); tree2.add(0); tree2.add(0); tree2.add(0)
idx, pri, _ = tree2.sample(3)
assert len(idx) == 3

# Test PrioritizedReplayBuffer
buf = PrioritizedReplayBuffer(100, obs_shape=(4, 84, 84), alpha=0.6)
for _ in range(50):
    buf.push(
        np.zeros((4, 84, 84), dtype=np.float32),
        1, 0.0,
        np.zeros((4, 84, 84), dtype=np.float32),
        False
    )
s, a, r, s2, d, w, idx = buf.sample(4)
assert s.shape == (4, 4, 84, 84)
assert w.shape == (4,)
assert w.max() <= 1.0

# Simulate TD errors and update
td = np.random.rand(4) * 10
buf.update_priorities(idx, td)
print(f"max_priority after update: {buf.max_priority:.2f}")

# Test beta schedule
buf2 = PrioritizedReplayBuffer(100, (4, 84, 84), beta=0.4, beta_end=1.0, beta_frac=0.5)
assert abs(buf2._current_beta() - 0.4) < 1e-3
buf2.update_beta_on_step(5_000_000)
assert abs(buf2._current_beta() - 1.0) < 1e-3

print("All smoke tests passed!")
"""