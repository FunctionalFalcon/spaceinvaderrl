import torch
import torch.nn as nn


# Q(s, ·; θ)
# literally the 2015 design of NatureCNN (Mnih et al. 2015).

class QNetwork(nn.Module):
  def __init__(self, num_actions: int):
    super().__init__()
    # Input: 4x84x84 (4 stacked grayscale frames like the paper)
    self.conv = nn.Sequential(
      nn.Conv2d(4, 32, kernel_size = 8, stride = 4), nn.ReLU(), # 32x20x20
      # (84 - 8) / 4 + 1 = 20
      nn.Conv2d(32, 64, kernel_size = 4, stride = 2), nn.ReLU(), # 64x9x9
      nn.Conv2d(64, 64, kernel_size = 3, stride = 1), nn.ReLU(), # 64x7x7
    )
    self.head = nn.Sequential(
      nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
      nn.Linear(512, num_actions),
      # output 1 Q-value per discrete action
    )
  

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """x: (B, 4, 84, 84) float32 in [0, 1]. Returns (B, |A|)."""
    return self.head(self.conv(x).flatten(1))


"""
m = QNetwork(num_actions=6)
x = torch.zeros(1, 4, 84, 84)
assert m(x).shape == (1, 6)
m(torch.zeros(32, 4, 84, 84))  # also test batched
print(sum(p.numel() for p in m.parameters()))  # 1,686,758

uncomment for testing
"""
