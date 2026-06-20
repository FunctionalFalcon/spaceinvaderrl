"""Record a video of the trained DQN agent playing Space Invaders.

Wraps the env with RecordVideo so each episode is saved as an .mp4 in
./videos/. The agent picks the best action (deterministic=True) so the
video shows its learned policy, not exploration noise.

Usage:
    python play.py                              # records 1 episode of dqn_final
    python play.py --episodes 3                 # more episodes for the demo
    python play.py --checkpoint <path>          # record a different checkpoint
"""
from __future__ import annotations

import argparse
import os
import sys

from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo

from preprocessing import make_env


_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(_HERE, "checkpoints", "dqn_final")
VIDEO_DIR = os.path.join(_HERE, "videos")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record a video of a trained agent.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT,
                   help="Path to the .zip model (no extension).")
    p.add_argument("--episodes", type=int, default=1,
                   help="Number of episodes to record (default 1).")
    p.add_argument("--name-prefix", type=str, default="trained",
                   help="Filename prefix for the mp4 (default 'trained').")
    p.add_argument("--seed", type=int, default=42,
                   help="Env seed for reproducibility.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.checkpoint + ".zip"):
        print(f"ERROR: checkpoint not found: {args.checkpoint}.zip")
        return 1

    os.makedirs(VIDEO_DIR, exist_ok=True)

    print("=== Space Invaders DQN Video Recorder ===")
    print(f"  Checkpoint  : {args.checkpoint}.zip")
    print(f"  Episodes    : {args.episodes}")
    print(f"  Output dir  : {VIDEO_DIR}")
    print(f"  Filename    : {args.name_prefix}-episode-N.mp4\n")

    # render_mode must be "rgb_array" so RecordVideo can grab frames.
    env = make_env(seed=args.seed, render_mode="rgb_array")

    # RecordVideo triggers on episode_number % episode_trigger == 0.
    # With lambda i: True, we record every episode.
    env = RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda i: True,
        name_prefix=args.name_prefix,
    )

    print("Loading model...")
    model = DQN.load(args.checkpoint, env=env, device="cpu")

    print(f"Recording {args.episodes} episode(s)...")
    for ep in range(args.episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False
        while not done:
            # deterministic=True -> argmax(Q(s, .)), i.e. the agent's best action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            steps += 1
            done = terminated or truncated
        print(f"  Episode {ep + 1}: {steps} steps, reward = {episode_reward:.1f}")

    env.close()
    print(f"\n[done] Videos saved to: {VIDEO_DIR}")
    print(f"        Open {args.name_prefix}-episode-0.mp4 in any video player.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
