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
    """Binary tree for prioritized sampling. All operations are O(log N).

    Uses a perfect binary tree padded to the next power of 2. Leaves occupy
    indices [_offset, _offset + capacity - 1]. The rest of the tree
    (indices [1, _offset-1]) stores partial sums for O(1) total retrieval
    and O(log N) search.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Pad to next power of 2 >= capacity. A perfect binary tree with N
        # leaves has exactly N-1 internal nodes (indices 0..N-1), but heap
        # child indexing (left=2*i, right=2*i+1) needs N slots for the leaves
        # as well, so total slots = 2*N. For capacity=4: padded=4, slots=8.
        # For capacity=1: padded=1, slots=2.
        self._padded = 1
        while self._padded < capacity:
            self._padded <<= 1
        # _tree[0] is unused (kept as 0) so child indexing works cleanly.
        # _tree[1.._padded-1] are internal sum nodes.
        # _tree[_padded..2*_padded-1] are the leaves.
        self._tree = np.zeros(2 * self._padded, dtype=np.float32)
        self._offset = self._padded  # first leaf index
        self._size = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, priority: float) -> int:
        """Add a new entry. Returns leaf index (0..capacity-1).

        When the ring buffer cycles, overwrites the oldest leaf.
        """
        leaf_idx = self._size % self.capacity
        tree_idx = self._offset + leaf_idx
        self._tree[tree_idx] = priority
        self._propagate(tree_idx)
        self._size = min(self._size + 1, self.capacity)
        return leaf_idx

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample `batch_size` leaf indices proportional to their priority."""
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float32)
        tree_indices = np.empty(batch_size, dtype=np.int64)

        total = self._tree[1]  # root at index 1
        if total == 0 or self._size == 0:
            rng = np.random.randint(0, max(1, self._size), size=batch_size)
            for i, r in enumerate(rng):
                ti = self._offset + r
                indices[i] = r
                priorities[i] = self._tree[ti]
                tree_indices[i] = ti
            return indices, priorities, tree_indices

        for i in range(batch_size):
            # Uniform offset within segment i of batch_size
            p = np.random.uniform(0, total)
            tree_idx = self._find(1, p)
            # Clamp: tree indices beyond the valid leaf range (padded region)
            # can survive _find's `idx < _offset` guard. Force them into the
            # last real slot so buffer indexing never overflows.
            max_valid_leaf = min(self._size, self.capacity) - 1
            leaf_idx = min(tree_idx - self._offset, max_valid_leaf)
            tree_idx = self._offset + leaf_idx
            indices[i] = leaf_idx
            priorities[i] = self._tree[tree_idx]
            tree_indices[i] = tree_idx
        return indices, priorities, tree_indices

    def update(self, tree_indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update leaf priorities. O(k log N) for k updates."""
        for ti, p in zip(tree_indices, priorities):
            self._tree[ti] = max(1e-10, float(p))
            self._propagate(ti)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, val: int) -> None:
        self._size = val

    def _total(self) -> float:
        return self._tree[1]

    def __repr__(self):
        return (f"SumTree(capacity={self.capacity}, padded={self._padded}, "
                f"total={self._tree[1]:.4f}, size={self._size})")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _propagate(self, idx: int) -> None:
        """Propagate a leaf change up to the root."""
        while idx > 1:
            idx //= 2
            self._tree[idx] = self._tree[2 * idx] + self._tree[2 * idx + 1]

    def _find(self, idx: int, p: float) -> int:
        """Walk down from internal node idx to the leaf containing priority p.

        At each internal node, go left if p falls in the left subtree,
        otherwise subtract left's sum and go right.
        """
        while idx < self._offset:
            left = 2 * idx
            if p < self._tree[left]:
                idx = left
            else:
                p -= self._tree[left]
                idx = left + 1
        # Hard cap: _find can wander into padded leaves (index >= _offset + capacity)
        # when the tree is right-heavy and the right branch extends beyond real slots.
        # Force any overshoot back to the last valid leaf so buffer indexing is safe.
        max_leaf = self._offset + min(self._size, self.capacity) - 1
        if idx > max_leaf:
            idx = max_leaf
        return idx


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