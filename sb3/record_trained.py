"""Record a video of the TRAINED DQN agent playing Space Invaders.

Bypasses gymnasium's RecordVideo (which has a fps=None crash with moviepy
on Kaggle) and writes .mp4 directly via imageio, which is preinstalled.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import DQN

from shared.preprocessing import make_env


HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(HERE, "videos")


def find_fallback_checkpoint(explicit: str) -> str:
    """Return the freshest available checkpoint, falling back from `explicit`.

    Priority: crashsave > final > highest-step resume_*.zip.
    """
    candidates: list[tuple[int, str]] = []
    for d in (
        os.path.dirname(explicit),
        HERE,
        "/kaggle/working/runs",
        "/kaggle/working/spaceinvaderrl_runs",
    ):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            full = os.path.join(d, name)
            if name == "dqn_fixed_crashsave.zip":
                candidates.append((10**12, full[:-4]))
            elif name == "dqn_fixed_final.zip":
                candidates.append((10**9, full[:-4]))
            else:
                m = re.match(r"dqn_fixed_resume_(\d+)_steps\.zip$", name)
                if m:
                    candidates.append((int(m.group(1)), full[:-4]))
    return max(candidates)[1] if candidates else explicit


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(HERE, "checkpoints", "dqn_final"))
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--name-prefix", type=str, default="trained")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=30,
                   help="Output video fps. Imageio doesn't care about env "
                        "metadata, so this is the only place fps lives.")
    args = p.parse_args()

    os.makedirs(VIDEO_DIR, exist_ok=True)

    ckpt = args.checkpoint
    if not os.path.exists(ckpt + ".zip"):
        print(f"[record_trained] {ckpt}.zip not found, searching fallbacks...")
        ckpt = find_fallback_checkpoint(ckpt)
        print(f"[record_trained] Using: {ckpt}.zip")

    print("=== Space Invaders TRAINED Agent Recorder (imageio) ===")
    print(f"  Checkpoint : {ckpt}.zip")
    print(f"  Episodes   : {args.episodes}")
    print(f"  FPS        : {args.fps}")
    print(f"  Output dir : {VIDEO_DIR}\n")

    env = make_env(seed=args.seed, render_mode="rgb_array")
    model = DQN.load(ckpt, env=env, device=args.device)

    for ep in range(args.episodes):
        out_path = os.path.join(VIDEO_DIR, f"{args.name_prefix}-episode-{ep}.mp4")
        print(f"  Recording episode {ep + 1} -> {out_path}")
        obs, info = env.reset()
        total = 0.0
        steps = 0
        done = False
        with imageio.get_writer(out_path, fps=args.fps) as writer:
            while not done:
                frame = env.render()  # rgb_array from preprocessing pipeline
                writer.append_data(frame)
                action, _ = model.predict(obs, deterministic=True)
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
