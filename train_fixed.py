"""Train a DQN agent on Space Invaders WITH the MinActionRepeat fix.

Same hyperparameters as train.py, but uses env_fixed() instead of make_env()
so the trained agent must commit to actions for >= min_repeat consecutive
agent-decisions. This breaks the runaway-LEFT failure mode (see
docs/theory.md §11.1 for the original bug, and §11.2 for the fix results).

Run from the project root:
    python train_fixed.py
"""
from __future__ import annotations

import argparse
import os
import re

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from preprocessing import env_fixed


# Resolve paths relative to this script (not the launch CWD)
_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_HERE, "checkpoints_fixed")
LOG_DIR = os.path.join(_HERE, "logs_fixed")


# ---------------------------------------------------------------------------
# Helpers (mirrors train.py so the two entry points feel consistent)
# ---------------------------------------------------------------------------

def list_checkpoints() -> list[tuple[str, int]]:
    """Return [(filename, step_count), ...] for every dqn_*.zip in CKPT_DIR.

    Files without a parseable step count (e.g. dqn_fixed_final.zip) are
    reported with step_count = -1 so they sort to the bottom.
    """
    if not os.path.isdir(CKPT_DIR):
        return []
    out: list[tuple[str, int]] = []
    for name in sorted(os.listdir(CKPT_DIR)):
        if not (name.startswith("dqn_fixed_") and name.endswith(".zip")):
            continue
        m = re.search(r"_(\d+)_steps", name)
        steps = int(m.group(1)) if m else -1
        out.append((name, steps))
    return out


def print_checkpoint_menu() -> None:
    checkpoints = list_checkpoints()
    print("Available fixed-env checkpoints in ./checkpoints_fixed:")
    if not checkpoints:
        print("  (none found)")
    else:
        checkpoints.sort(key=lambda x: (x[1] < 0, -x[1]))
        for fname, steps in checkpoints:
            if steps >= 0:
                print(f"  - {fname:<40s}  ({steps:,} steps)")
            else:
                print(f"  - {fname:<40s}  (final)")
    print()


def find_latest_resume_checkpoint() -> str | None:
    """Find the highest-step dqn_fixed_resume_*.zip file in CKPT_DIR."""
    if not os.path.isdir(CKPT_DIR):
        return None
    pattern = re.compile(r"dqn_fixed_resume_(\d+)_steps\.zip$")
    best: tuple[int, str] | None = None
    for name in os.listdir(CKPT_DIR):
        m = pattern.match(name)
        if not m:
            continue
        steps = int(m.group(1))
        if best is None or steps > best[0]:
            best = (steps, os.path.join(CKPT_DIR, name))
    return best[1] if best else None


def prompt_choice() -> str:
    print("What do you want to do?")
    print("  [1] Train from scratch (fresh, 300k steps)")
    print("  [2] Resume from a fixed-env checkpoint")
    print("  [3] Run quick smoke test (5,000 steps, fresh)")
    print("  [4] Exit")
    while True:
        choice = input("Choice: ").strip()
        if choice in {"1", "2", "3", "4"}:
            return choice
        print("Please enter 1, 2, 3, or 4.")


def prompt_checkpoint(checkpoints: list[tuple[str, int]]) -> str:
    """Ask the user which checkpoint to resume from. Returns full path."""
    if not checkpoints:
        print("No checkpoints found. Starting fresh instead.")
        return ""
    print("Which checkpoint?")
    for i, (fname, steps) in enumerate(checkpoints, start=1):
        if steps >= 0:
            print(f"  [{i}] {fname}  ({steps:,} steps)")
        else:
            print(f"  [{i}] {fname}")
    while True:
        raw = input(f"Choice [1-{len(checkpoints)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(checkpoints):
            chosen = checkpoints[int(raw) - 1][0]
            return os.path.join(CKPT_DIR, chosen)
        print("Invalid choice, try again.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    mode: str,                       # "scratch" | "resume" | "smoketest"
    resume_path: str = "",
    total_timesteps: int = 200_000,
    ckpt_dir: str | None = None,
    log_dir: str | None = None,
    save_freq: int = 25_000,
    device: str = "cpu",
    buffer_size: int | None = None,
    learning_starts: int | None = None,
) -> None:
    """Train (or resume) the DQN agent on the MinActionRepeat-wrapped env.

    ckpt_dir / log_dir: override the default ./checkpoints_fixed and
        ./logs_fixed paths. Use this on Colab/Kaggle to point at a mounted
        Drive or /kaggle/working so artifacts survive session restarts.
    save_freq: steps between CheckpointCallback saves. Lower = more resume
        points but more disk usage; 50_000 is a safe default for 1M-step runs.
    device: "cpu" or "cuda" — flip to "cuda" on Colab/Kaggle for ~10x speedup.
    buffer_size / learning_starts: optional overrides; default to mode-tuned
        values below.
    """
    if ckpt_dir is None:
        ckpt_dir = CKPT_DIR
    if log_dir is None:
        log_dir = LOG_DIR

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = make_vec_env(env_fixed, n_envs=1, seed=0)

    if mode == "resume":
        if not resume_path:
            # Fall back to the legacy "./checkpoints_fixed" location so
            # existing runs still work without --resume.
            legacy_pattern = re.compile(r"dqn_fixed_resume_(\d+)_steps\.zip$")
            best: tuple[int, str] | None = None
            for d in (ckpt_dir, CKPT_DIR):
                if not os.path.isdir(d):
                    continue
                for name in os.listdir(d):
                    m = legacy_pattern.match(name)
                    if not m:
                        continue
                    steps = int(m.group(1))
                    if best is None or steps > best[0]:
                        best = (steps, os.path.join(d, name))
            resume_path = best[1] if best else ""
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            model = DQN.load(
                resume_path,
                env=env,
                reset_num_timesteps=True,
                tensorboard_log=log_dir,
                device=device,
            )
        else:
            print("No resumable checkpoint found; falling back to fresh training.")
            mode = "scratch"

    if mode != "resume":
        # Same hyperparameters as train.py for a fair before/after comparison.
        # Only the env wrapper changes. CLI flags override mode-based defaults
        # so Colab runs can use a larger replay buffer.
        if buffer_size is None:
            buffer_size = 10_000 if mode == "smoketest" else 20_000
        if learning_starts is None:
            learning_starts = 1_000 if mode == "smoketest" else 5_000
        print(f"Starting fresh training "
              f"(buffer={buffer_size:,}, learning_starts={learning_starts:,}, "
              f"device={device}).")

        model = DQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            max_grad_norm=10.0,
            tensorboard_log=log_dir,
            device=device,
            policy_kwargs={"normalize_images": False},
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=ckpt_dir,
        name_prefix="dqn_fixed_resume",
        save_replay_buffer=False,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        tb_log_name="dqn_fixed_full",
        progress_bar=True,
    )

    final_path = os.path.join(ckpt_dir, "dqn_fixed_final")
    model.save(final_path)
    env.close()
    print(f"Done. Saved {final_path}.zip")


def main() -> None:
    """Two entry paths:

    1. **Interactive** (no CLI flags): original menu, prompts for choice and
       checkpoint. Used on a local machine.

    2. **Non-interactive** (any CLI flag passed): skips the menu and runs
       straight through. Used on Colab/Kaggle where `input()` would hang.

    Examples:
        # 1M steps from scratch on a GPU, save to Drive every 50k:
        python train_fixed.py --timesteps 1000000 --device cuda \\
            --save-dir /content/drive/MyDrive/spaceinvaderrl_runs \\
            --save-freq 50000

        # Resume the latest checkpoint for another 500k steps:
        python train_fixed.py --resume latest --timesteps 500000 --device cuda

        # Quick smoke test (10k steps, CPU):
        python train_fixed.py --smoketest
    """
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on Space Invaders (MinActionRepeat fix).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total environment steps to train for.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Torch device. Use 'cuda' on Colab/Kaggle.")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override checkpoint directory. "
                             "Defaults to ./checkpoints_fixed.")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Override tensorboard log directory. "
                             "Defaults to ./logs_fixed.")
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Steps between checkpoint saves.")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer size. Default 20k (smoketest 10k).")
    parser.add_argument("--learning-starts", type=int, default=None,
                        help="Random steps before training starts. "
                             "Default 5k (smoketest 1k).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .zip checkpoint to resume from, or "
                             "'latest' to auto-pick the highest-step one in "
                             "--save-dir.")
    parser.add_argument("--smoketest", action="store_true",
                        help="Run a 10k-step smoke test (sanity check).")
    args = parser.parse_args()

    # If the user passed ANY flag, we're in non-interactive mode.
    non_interactive = any(
        v is not None and v is not False
        for v in vars(args).values()
    )

    if not non_interactive:
        # Legacy interactive menu (local-machine flow, unchanged behavior).
        print("=== Space Invaders DQN Training (MinActionRepeat fix) ===\n")
        print_checkpoint_menu()
        choice = prompt_choice()
        if choice == "4":
            print("Bye.")
            return
        if choice == "1":
            train(mode="scratch", total_timesteps=1_000_000)
        elif choice == "2":
            checkpoints = [c for c in list_checkpoints() if c[1] >= 0]
            if not checkpoints:
                print("No resumable checkpoints (with step counts) found. "
                      "Train from scratch first.")
                return
            checkpoints.sort(key=lambda x: x[1])
            chosen = prompt_checkpoint(checkpoints)
            train(mode="resume", resume_path=chosen, total_timesteps=1_000_000)
        elif choice == "3":
            train(mode="smoketest", total_timesteps=10_000)
        return

    # Non-interactive path (Colab / Kaggle).
    if args.smoketest:
        train(
            mode="smoketest",
            total_timesteps=10_000,
            ckpt_dir=args.save_dir,
            log_dir=args.log_dir,
            save_freq=args.save_freq,
            device=args.device,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
        )
        return

    if args.resume:
        resume_path = args.resume
        if resume_path == "latest":
            resume_path = None  # let train() auto-discover
        train(
            mode="resume",
            resume_path=resume_path or "",
            total_timesteps=args.timesteps,
            ckpt_dir=args.save_dir,
            log_dir=args.log_dir,
            save_freq=args.save_freq,
            device=args.device,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
        )
    else:
        train(
            mode="scratch",
            total_timesteps=args.timesteps,
            ckpt_dir=args.save_dir,
            log_dir=args.log_dir,
            save_freq=args.save_freq,
            device=args.device,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
        )


if __name__ == "__main__":
    main()
