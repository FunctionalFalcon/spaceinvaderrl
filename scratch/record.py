"""Record a video of the FROM-SCRATCH DQN agent playing Space Invaders.

Loads a torch .pt checkpoint, rebuilds QNetwork, and writes an MP4 via
imageio. Designed to be run from the project root on a local box.

Usage:
    python record_local.py --checkpoint scratch/spaceinvaderrl/runs/dqn_scratch_final.pt
    python record_local.py --checkpoint scratch/spaceinvaderrl/runs/dqn_scratch_step_2000000.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Insert the project root at the front of sys.path so the absolute imports
# below (`from scratch.X`, `from shared.X`) resolve regardless of CWD.
# Project root is the parent of this script's directory (scratch/).
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import imageio.v2 as imageio
import torch

from scratch.network import QNetwork
from scratch.hyperparam import Hyperparameters
from scratch.evaluate import greedy_action
from shared.preprocessing import env_fixed

VIDEO_DIR = _PROJECT_ROOT / "videos"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the .pt checkpoint.")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--name-prefix", type=str, default="trained-scratch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        return 1

    VIDEO_DIR.mkdir(exist_ok=True)

    print("=== From-Scratch DQN Trained Agent Recorder (imageio) ===")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  FPS        : {args.fps}")
    print(f"  Output dir : {VIDEO_DIR}\n")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = Hyperparameters(**ckpt.get("hp", {}))
    num_actions = int(getattr(hp, "num_actions", 6))
    min_repeat = int(getattr(hp, "min_repeat", 4))
    q_net = QNetwork(num_actions=num_actions)
    q_net.load_state_dict(ckpt["model_state_dict"])
    q_net.eval()

    env = env_fixed(seed=args.seed, render_mode="rgb_array", min_repeat=min_repeat)

    for ep in range(args.episodes):
        out_path = VIDEO_DIR / f"{args.name_prefix}-episode-{ep}.mp4"
        print(f"  Recording episode {ep + 1} -> {out_path}")
        obs, _ = env.reset()
        total = 0.0
        steps = 0
        done = False
        with imageio.get_writer(out_path, fps=args.fps) as writer:
            while not done:
                writer.append_data(env.render())
                action = greedy_action(q_net, obs, device="cpu")
                obs, r, term, trunc, _ = env.step(action)
                total += float(r)
                steps += 1
                done = term or trunc
        print(f"  Episode {ep + 1}: {steps} steps, reward = {total:.1f}")
        print(f"    saved: {out_path}")

    env.close()
    print(f"\n[done] Videos saved to: {VIDEO_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())