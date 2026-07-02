from dataclasses import dataclass, field


@dataclass
class Hyperparameters:
 """Hyperparameters for Dueling DQN with ε-greedy exploration and Prioritized Replay.

 Mirrors Mnih 2015 defaults with Rainbow-era improvements:
 - Dueling DQN: V(s) and A(s,a) streams (Wang et al., 2016)
 - ε-greedy exploration (Mnih 2015) — Noisy Nets removed after causing PER oscillation
 - Prioritized Replay: smart sampling by TD error (Schaul et al., 2015)
 - Soft target updates (tau=0.005) + hard reset every 1M steps

 `device` is auto-detected at train time, but stored here as a default so
 the dataclass can be pickled/serialized cleanly without importing torch.
 """
 env_id: str = "ALE/SpaceInvaders-v5"
 seed: int = 0
 total_steps: int = 5_000_000   # Longer run for Rainbow-level improvements
 buffer_size: int = 100_000     # Larger buffer for prioritized replay (prioritized needs more buffer)
 learning_starts: int = 5_000
 batch_size: int = 32
 gamma: float = 0.99
 lr: float = 1e-4
 target_update_tau: float = 0.005   # Soft update coefficient (online -> target). 0 = hard copy. ~0.01 = Rainbow default
 target_hard_reset_freq: int = 1_000_000   # Hard-copy online->target every N steps to fully break drift (0=disabled)
 save_freq: int = 100_000
 eval_freq: int = 100_000
 eval_episodes: int = 10
 eps_start: float = 1.0
 eps_end: float = 0.01
 eps_frac: float = 0.15
 max_grad_norm: float = 10.0
 min_repeat: int = 3
 device: str = "auto" # "auto" | "cpu" | "cuda". Resolved at train time.
 num_actions: int = 6
 auto_zip_freq: int = 200_000 # Steps between rescue-zip writes (0 = disabled).
 auto_zip_keep: int = 5 # How many recent step checkpoints to include in zip.
 # Prioritized Replay parameters (Schaul et al., 2015)
 prio_alpha: float = 0.4  # Prioritization exponent: 0=uniform, 1=pure TD-error. 0.4 = less aggressive than 0.6, more stable
 prio_beta: float = 0.4   # Initial IS weight exponent: 0=no correction, 1=full correction
 prio_beta_end: float = 1.0
 prio_beta_frac: float = 0.75  # Fraction of training over which beta anneals to prio_beta_end
 use_prioritized: bool = True       # True = prioritized replay, False = uniform
