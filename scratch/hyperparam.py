from dataclasses import dataclass, field


@dataclass
class Hyperparameters:
 """Hyperparameters for from-scratch DQN.

 Mirrors Mnih 2015 / SB3 defaults for the field we kept, with three
 intentional deviations for our setup:
 - total_steps: 1.2M -> 2M (better chance of matching SB3 baseline 304.2)
 - eps_frac: 0.10 -> 0.15 (slightly slower epsilon decay, more exploration)
 - min_repeat: 4 -> 3 (faster dodge response than the SB3-fixed run)

 `device` is auto-detected at train time, but stored here as a default so
 the dataclass can be pickled/serialized cleanly without importing torch.
 """
 env_id: str = "ALE/SpaceInvaders-v5"
 seed: int = 0
 total_steps: int = 2_000_000
 buffer_size: int = 20_000
 learning_starts: int = 5_000
 batch_size: int = 32
 gamma: float = 0.99
 lr: float = 1e-4
 target_update: int = 1_000
 save_freq: int = 50_000
 eval_freq: int = 50_000
 eval_episodes: int = 5
 eps_start: float = 1.0
 eps_end: float = 0.01
 eps_frac: float = 0.15
 max_grad_norm: float = 10.0
 min_repeat: int = 3
 device: str = "auto" # "auto" | "cpu" | "cuda". Resolved at train time.
 num_actions: int = 6
 auto_zip_freq: int = 200_000 # Steps between rescue-zip writes (0 = disabled).
 auto_zip_keep: int = 5 # How many recent step checkpoints to include in zip.
