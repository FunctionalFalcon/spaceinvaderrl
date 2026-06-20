"""Train a DQN agent on Space Invaders WITH the MinActionRepeat fix.

Same hyperparameters as train.py, but uses env_fixed() instead of make_env()
so the trained agent must commit to actions for >= min_repeat consecutive
agent-decisions. This breaks the runaway-LEFT failure mode (see
docs/theory.md §11.1 for the original bug, and §11.2 for the fix results).

Run from the project root:
    python train_fixed.py
"""
from __future__ import annotations

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
) -> None:
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = make_vec_env(env_fixed, n_envs=1, seed=0)

    if mode == "resume":
        if not resume_path:
            resume_path = find_latest_resume_checkpoint() or ""
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            model = DQN.load(
                resume_path,
                env=env,
                reset_num_timesteps=True,
                tensorboard_log=LOG_DIR,
                device="cpu",
            )
        else:
            print("No resumable checkpoint found; falling back to fresh training.")
            mode = "scratch"

    if mode != "resume":
        # Same hyperparameters as train.py for a fair before/after comparison.
        # Only the env wrapper changes.
        buffer_size = 10_000 if mode == "smoketest" else 20_000
        learning_starts = 1_000 if mode == "smoketest" else 5_000
        print(f"Starting fresh training "
              f"(buffer={buffer_size:,}, learning_starts={learning_starts:,}).")

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
            tensorboard_log=LOG_DIR,
            device="cpu",
            policy_kwargs={"normalize_images": False},
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=CKPT_DIR,
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

    final_path = os.path.join(CKPT_DIR, "dqn_fixed_final")
    model.save(final_path)
    env.close()
    print(f"Done. Saved {final_path}.zip")


def main() -> None:
    print("=== Space Invaders DQN Training (MinActionRepeat fix) ===\n")
    print_checkpoint_menu()
    choice = prompt_choice()

    if choice == "4":
        print("Bye.")
        return

    if choice == "1":
        train(mode="scratch", total_timesteps=300_000)

    elif choice == "2":
        checkpoints = [c for c in list_checkpoints() if c[1] >= 0]
        if not checkpoints:
            print("No resumable checkpoints (with step counts) found. "
                  "Train from scratch first.")
            return
        checkpoints.sort(key=lambda x: x[1])
        chosen = prompt_checkpoint(checkpoints)
        train(mode="resume", resume_path=chosen, total_timesteps=300_000)

    elif choice == "3":
        train(mode="smoketest", total_timesteps=5_000)


if __name__ == "__main__":
    main()
