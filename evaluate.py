"""Quantitative evaluation of a trained DQN agent.

Runs the trained agent for N episodes and reports mean +/- std episode reward.
This is the number you'll quote in your presentation.

Usage:
    python evaluate.py                        # uses checkpoints/dqn_final.zip, 20 episodes
    python evaluate.py --episodes 50          # more episodes = more stable estimate
    python evaluate.py --checkpoint <path>    # evaluate a specific checkpoint
"""
from __future__ import annotations

import argparse
import os
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from preprocessing import make_env


_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(_HERE, "checkpoints", "dqn_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained DQN agent.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT,
                   help="Path to the .zip model to evaluate (no extension).")
    p.add_argument("--episodes", type=int, default=20,
                   help="Number of episodes to run (default 20).")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use greedy action selection (default: on).")
    p.add_argument("--seed", type=int, default=12345,
                   help="Env seed for reproducibility.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.checkpoint + ".zip"):
        print(f"ERROR: checkpoint not found: {args.checkpoint}.zip")
        print("Run `python train.py` first, or pass --checkpoint <path>.")
        return 1

    print("=== Space Invaders DQN Evaluation ===")
    print(f"  Checkpoint : {args.checkpoint}.zip")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Determinism: {args.deterministic}")
    print(f"  Seed       : {args.seed}\n")

    # Build env WITHOUT sticky actions noise so the eval is reproducible
    # (training uses p=0.25; we drop it here for a clean number).
    # We keep the same preprocessing (grayscale, 84x84, 4-frame stack) so
    # the observation the network sees is identical to training.
    env = make_env(seed=args.seed, render_mode=None)

    print("Loading model...")
    model = DQN.load(args.checkpoint, env=env, device="cpu")

    print(f"Running {args.episodes} evaluation episodes...")
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        deterministic=args.deterministic,
    )

    print()
    print("=" * 50)
    print(f"  Mean reward: {mean_reward:8.1f} +/- {std_reward:.1f}")
    print("=" * 50)
    print()
    print("Compare to a random baseline (~100-150) to show the agent learned.")
    print("Run `python record_random.py` first to get a random-baseline number.")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
