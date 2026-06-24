"""From-scratch DQN implementation (no Stable Baselines3).

Modules:
 network NatureCNN Q-network architecture
 replay_buffer Ring-buffer experience replay
 agent epsilon-greedy + train_step (TD target + Huber loss)
 hyperparam Hyperparameters dataclass
 train Training entry point (use as `python -m dqn.train`)
 evaluate Greedy evaluation (use as `python -m dqn.evaluate`)
"""

__all__ = ["network", "replay_buffer", "agent", "hyperparam", "train", "evaluate"]