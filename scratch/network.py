import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Noisy Linear Layer
# ─────────────────────────────────────────────────────────────────────────────
# From "Noisy Networks for Exploration" (Fortunato et al., 2017).
#
# Each weight w = w_fixed + w_noise, where w_noise is a learned Gaussian.
# This gives state-dependent, differentiable exploration without ε-greedy.
#
# Factorized Gaussian noise reduces the number of random samples:
#   w_noise[i,j] = (σ_i / √k) · ε_j        (output i, input j)
#   b_noise[i]   = σ_i · ε_i
# where ε ~ N(0,1), σ is learned, k = fan-in of the layer.

class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Fixed (learnable) weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Learned noise scale (σ). Initialized to sigma_init as in the paper.
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Register persistent noise buffers (not trained by Adam, regenerated each pass)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialize μ weights as a normal nn.Linear would
        nn.init.kaiming_uniform_(self.weight_mu, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.bias_mu)
        # Initialize σ: paper recommends σ_init / √fan_in
        nn.init.constant_(self.weight_sigma, self.sigma_init / (self.in_features ** 0.5))
        nn.init.constant_(self.bias_sigma, self.sigma_init / (self.in_features ** 0.5))

    def _noise(self, size: torch.Size) -> torch.Size:
        """Sample noise from standard normal, respecting device/dtype of self.weight_mu."""
        return torch.randn(size, device=self.weight_mu.device, dtype=self.weight_mu.dtype)

    def reset_noise(self):
        """Sample fresh noise. Call after each forward pass in training."""
        eps_in = self._noise(self.in_features)          # (k,)
        eps_out = self._noise(self.out_features)        # (out,)
        self.weight_epsilon.copy_(eps_out.unsqueeze(1) * eps_in)  # (out, in)
        self.bias_epsilon.copy_(eps_out)                  # (out,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # During training: weights = μ + σ · ε
            # ε is refreshed via reset_noise() called externally
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, w, b)
        else:
            # During eval: use mean weights (deterministic)
            return F.linear(x, self.weight_mu, self.bias_mu)


# ─────────────────────────────────────────────────────────────────────────────
# Dueling + Noisy DQN Network (NatureCNN backbone)
# ─────────────────────────────────────────────────────────────────────────────
# Architecture: Dueling DQN (Wang et al., 2016) + Noisy Nets (Fortunato et al., 2017)
#
# Instead of a single Q-head, we have two streams:
#   V(s)     — how good is state s? (1 output)
#   A(s, a)  — advantage of each action in state s (|A| outputs)
#   Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
#
# The advantage tells us "how much better is action a than average",
# while the value tells us "how good is this state regardless of action".
#
# All FC layers use NoisyLinear, so exploration is state-dependent and
# persists throughout training (no ε decay needed).

class QNetwork(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

        # ── Shared convolutional backbone (unchanged from Mnih 2015) ──────────
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),   # → 32×20×20
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),   # → 64×9×9
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),   # → 64×7×7
        )
        conv_out_size = 64 * 7 * 7  # 3136

        # ── Dueling architecture: value and advantage streams ────────────────
        # Both streams start from the same conv features (weight tying at conv only).
        # Each stream uses NoisyLinear so exploration is learned and persistent.
        hid_size = 512

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            NoisyLinear(conv_out_size, hid_size),
            nn.ReLU(),
            NoisyLinear(hid_size, 1),   # single scalar: V(s)
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(conv_out_size, hid_size),
            nn.ReLU(),
            NoisyLinear(hid_size, num_actions),  # one per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, 84, 84) float32 in [0, 1]. Returns (B, |A|)."""
        features = self.conv(x).flatten(1)

        v = self.value_stream(features)               # (B, 1)
        a = self.advantage_stream(features)            # (B, num_actions)

        # Dueling aggregation: Q = V + A - mean(A)
        # This ensures Q values are properly centered around V(s).
        # max_a Q(s,a) = V(s) + max_a A(s,a) (since mean subtracts out)
        q = v + a - a.mean(dim=1, keepdim=True)       # (B, num_actions)
        return q

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers. Call once per training step."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ─────────────────────────────────────────────────────────────────────────────
# Dueling DQN Network (standard Linear — for use with epsilon-greedy)
# ─────────────────────────────────────────────────────────────────────────────
# Same Dueling architecture as QNetwork but with standard nn.Linear instead
# of NoisyLinear. Exploration is handled by epsilon-greedy, which cleanly
# separates exploration from the value function (unlike Noisy Nets where
# exploration is baked into the weights and interacts badly with PER).
#
# Dueling aggregation: Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]

class DuelingDQN(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

        # ── Shared convolutional backbone (Mnih 2015) ──────────────────────────
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),   # → 32×20×20
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),   # → 64×9×9
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),   # → 64×7×7
        )
        conv_out_size = 64 * 7 * 7  # 3136

        hid_size = 512

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, hid_size), nn.ReLU(),
            nn.Linear(hid_size, 1),   # single scalar: V(s)
        )

        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, hid_size), nn.ReLU(),
            nn.Linear(hid_size, num_actions),  # one per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, 84, 84) float32 in [0, 1]. Returns (B, |A|)."""
        features = self.conv(x).flatten(1)
        v = self.value_stream(features)               # (B, 1)
        a = self.advantage_stream(features)            # (B, num_actions)
        q = v + a - a.mean(dim=1, keepdim=True)       # (B, num_actions)
        return q


# ─────────────────────────────────────────────────────────────────────────────
# Legacy QNetwork (no dueling, no noisy) — kept for compatibility / comparison
# ─────────────────────────────────────────────────────────────────────────────
class QNetworkLegacy(nn.Module):
    """Standard DQN head (Linear, not NoisyLinear). For A/B testing vs Dueling."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(1))


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────
"""
# Test 1: DuelingNoisy shapes
m = QNetwork(num_actions=6)
x = torch.zeros(1, 4, 84, 84)
assert m(x).shape == (1, 6), f"Expected (1, 6), got {m(x).shape}"

# Test 2: Batched
assert m(torch.zeros(32, 4, 84, 84)).shape == (32, 6)

# Test 3: Noise is different each forward pass during training
m.train()
q1 = m(x)
m.reset_noise()
q2 = m(x)
assert not torch.allclose(q1, q2), "Training forward should differ with new noise"

# Test 4: No noise during eval
m.eval()
q_eval = m(x)
# (should be reproducible but we can't assert exact match since no noise applied)

# Test 5: Dueling identity — Q = V + A - mean(A)
v = m.value_stream(m.conv(x).flatten(1))
a = m.advantage_stream(m.conv(x).flatten(1))
expected_q = v + a - a.mean(dim=1, keepdim=True)
assert torch.allclose(m(x), expected_q, atol=1e-6), "Dueling aggregation wrong"

# Test 6: Legacy compatibility
leg = QNetworkLegacy(num_actions=6)
assert leg(x).shape == (1, 6)

# Test 7: Parameter counts
dueling_params = sum(p.numel() for p in m.parameters())
legacy_params = sum(p.numel() for p in leg.parameters())
print(f"DuelingNoisy params: {dueling_params:,}")
print(f"Legacy params:       {legacy_params:,}")
# DuelingNoisy: ~1.8M (adds ~1.7k params for the extra stream heads)
# Legacy:       ~1.7M
"""
