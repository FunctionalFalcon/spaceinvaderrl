"""Record a video of a RANDOM agent playing Space Invaders (baseline).

Bypasses gymnasium's RecordVideo (fps=None crash with moviepy on Kaggle)
and writes .mp4 directly via imageio.
"""
from __future__ import annotations

import argparse
import os
import sys

# When this file is run as `python sb3/record_random.py`, Python adds the
# script's directory (sb3/) to sys.path[0], not the repo root. The
# `from shared.preprocessing import make_env` below needs the repo root on
# sys.path. Insert the parent of this script's directory so absolute
# imports of `shared.X` and `scratch.X` resolve from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import imageio.v2 as imageio

from shared.preprocessing import make_env


HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(HERE, "videos")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--name-prefix", type=str, default="random")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    os.makedirs(VIDEO_DIR, exist_ok=True)
    print("=== Random baseline recorder (imageio) ===")
    print(f"  Episodes : {args.episodes}")
    print(f"  FPS      : {args.fps}")
    print(f"  Output   : {VIDEO_DIR}\n")

    env = make_env(seed=args.seed, render_mode="rgb_array")

    for ep in range(args.episodes):
        out_path = os.path.join(VIDEO_DIR, f"{args.name_prefix}-episode-{ep}.mp4")
        print(f"  Recording episode {ep + 1} -> {out_path}")
        obs, info = env.reset()
        total = 0.0
        steps = 0
        done = False
        with imageio.get_writer(out_path, fps=args.fps) as writer:
            while not done:
                frame = env.render()
                writer.append_data(frame)
                action = env.action_space.sample()
                obs, reward, term, trunc, _ = env.step(action)
                total += float(reward)
                steps += 1
                done = term or trunc
        print(f"  Episode {ep + 1}: {steps} steps, reward = {total:.1f}")
        print(f"    saved: {out_path}")

    env.close()
    print(f"\n[done] Videos saved to: {VIDEO_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
