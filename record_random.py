"""Record a video of a RANDOM agent playing Space Invaders.

Side-by-side comparison for the teacher demo: shows what 'before training'
looks like vs. the trained agent's video. A random agent will usually
miss most shots and die very quickly.

Usage:
    python record_random.py                     # records 1 random episode
    python record_random.py --episodes 3        # multiple random episodes
"""
from __future__ import annotations

import argparse
import os
import sys

from gymnasium.wrappers import RecordVideo

from preprocessing import make_env


HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(HERE, "videos")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record a video of a random agent.")
    p.add_argument("--episodes", type=int, default=1,
                   help="Number of episodes to record (default 1).")
    p.add_argument("--name-prefix", type=str, default="random",
                   help="Filename prefix for the mp4 (default 'random').")
    p.add_argument("--seed", type=int, default=42,
                   help="Env seed for reproducibility.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    os.makedirs(VIDEO_DIR, exist_ok=True)

    print("=== Space Invaders Random Baseline Recorder ===")
    print(f"  Episodes    : {args.episodes}")
    print(f"  Output dir  : {VIDEO_DIR}")
    print(f"  Filename    : {args.name_prefix}-episode-N.mp4\n")

    env = make_env(seed=args.seed, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda i: True,
        name_prefix=args.name_prefix,
    )

    print(f"Recording {args.episodes} random-action episode(s)...")
    for ep in range(args.episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False
        while not done:
            # uniform random over the 6 discrete actions
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            steps += 1
            done = terminated or truncated
        print(f"  Episode {ep + 1}: {steps} steps, reward = {episode_reward:.1f}")

    env.close()
    print(f"\n[done] Videos saved to: {VIDEO_DIR}")
    print(f"        Compare {args.name_prefix}-episode-0.mp4 to trained-episode-0.mp4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
