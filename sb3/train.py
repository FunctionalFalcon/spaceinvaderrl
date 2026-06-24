from __future__ import annotations

import os
import re
import time

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from shared.preprocessing import make_env


# Resolve paths relative to this script (not the launch CWD)
_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_HERE, "checkpoints")
LOG_DIR = os.path.join(_HERE, "logs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_checkpoints() -> list[tuple[str, int]]:
    """Return [(filename, step_count), ...] for every dqn_*.zip in CKPT_DIR.

    Files without a parseable step count (e.g. dqn_final.zip, dqn_smoketest.zip)
    are reported with step_count = -1 so they sort to the bottom.
    """
    if not os.path.isdir(CKPT_DIR):
        return []
    out: list[tuple[str, int]] = []
    for name in sorted(os.listdir(CKPT_DIR)):
        if not (name.startswith("dqn_") and name.endswith(".zip")):
            continue
        m = re.search(r"_(\d+)_steps", name)
        steps = int(m.group(1)) if m else -1
        out.append((name, steps))
    return out


def print_checkpoint_menu() -> None:
    checkpoints = list_checkpoints()
    print("Available checkpoints in ./checkpoints:")
    if not checkpoints:
        print("  (none found)")
    else:
        # Sort: known step counts first (descending), then unknowns
        checkpoints.sort(key=lambda x: (x[1] < 0, -x[1]))
        for fname, steps in checkpoints:
            if steps >= 0:
                print(f"  - {fname:<40s}  ({steps:,} steps)")
            else:
                print(f"  - {fname:<40s}  (final / smoketest)")
    print()


def find_latest_resume_checkpoint() -> str | None:
    """Find the highest-step dqn_resume_*.zip file in CKPT_DIR.

    Returns the full path to the latest checkpoint, or None if none exist.
    Used by the resume flow so the user doesn't have to type the step count.
    """
    if not os.path.isdir(CKPT_DIR):
        return None
    pattern = re.compile(r"dqn_resume_(\d+)_steps\.zip$")
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
    print("  [1] Train from scratch (fresh)")
    print("  [2] Resume from a checkpoint")
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
# Custom callback for the progress line
# ---------------------------------------------------------------------------

class ProgressLoggerCallback(BaseCallback):  # type: ignore[misc]
    """Print a one-line progress summary every `print_every` env steps."""

    def __init__(self, print_every: int = 20_000, verbose: int = 1):
        super().__init__(verbose)
        self.print_every = print_every
        self._next_print = print_every
        self._last_time = time.time()
        self._last_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_print:
            return True

        # Pull the recent-episode returns from SB3's internal buffer
        ep_buf = self.model.ep_info_buffer
        if ep_buf is not None and len(ep_buf) > 0:
            returns = [ep["r"] for ep in ep_buf]
            mean_return = float(np.mean(returns))
            n_window = len(returns)
        else:
            mean_return = 0.0
            n_window = 0

        now = time.time()
        sps = (self.num_timesteps - self._last_step) / max(now - self._last_time, 1e-6)
        loss = float(self.model.logger.name_to_value.get("train/loss", 0.0))
        eps = float(self.model.exploration_rate)  # type: ignore[attr-defined]

        # Show the rolling-100-ep mean; if fewer episodes completed, show actual count
        if n_window >= 100:
            window_label = "100ep"
        else:
            window_label = f"{n_window:>3}ep"
        print(
            f"  Step {self.num_timesteps:>9,} | "
            f"Episodes: {self.model._episode_num:>5} | "
            f"Mean return ({window_label}): "
            f"{mean_return:>7.1f} | "
            f"Loss: {loss:.4f} | "
            f"ε: {eps:.3f} | "
            f"SPS: {sps:.0f}"
        )

        self._last_time = now
        self._last_step = self.num_timesteps
        self._next_print += self.print_every
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    mode: str,                       # "scratch" | "resume" | "smoketest"
    resume_path: str = "",
    total_timesteps: int = 300_000,
) -> None:
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # SB3's built-in rollout table verbosity. Read once, up front, so we
    # can pass it to BOTH the fresh DQN constructor and the resume path.
    # Set SB3_VERBOSE=1 to re-enable the full rollout table; default is off
    # so the console only shows our clean single-line progress.
    sb3_verbose = int(os.environ.get("SB3_VERBOSE", "0"))
    print(f"SB3 rollout-table verbosity: {sb3_verbose} "
          f"(set SB3_VERBOSE=1 to enable the full table)\n")

    env = make_vec_env(make_env, n_envs=1, seed=0)

    # Auto-resume: if no explicit path was passed but a checkpoint exists,
    # pick the latest one automatically. This means resuming is a no-op
    # for the user — just choose [2] and we'll find the newest file.
    if mode == "resume":
        if not resume_path:
            resume_path = find_latest_resume_checkpoint() or ""
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            model = DQN.load(
                resume_path,
                env=env,
                reset_num_timesteps=True,    # counter restarts so progress lines are clean
                tensorboard_log=LOG_DIR,
                device="cpu",
                verbose=sb3_verbose,
            )
        else:
            print("No resumable checkpoint found; falling back to fresh training.")
            mode = "scratch"

    if mode != "resume":
        # Fresh agent (works for both "scratch" and "smoketest").
        # Buffer sizes are tuned to fit an 8 GB available-RAM budget
        # (each transition is ~330 KB: 4 frames * 84 * 84 * float32 * 3 arrays).
        # 20k = ~6.3 GB replay buffer; safe margin for Windows + apps.
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
            verbose=sb3_verbose,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=CKPT_DIR,
        name_prefix="dqn_resume",
        save_replay_buffer=False,
        verbose=1,
    )
    progress_cb = ProgressLoggerCallback(print_every=20_000, verbose=1)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, progress_cb],
        tb_log_name="dqn_full",
        progress_bar=True,
    )

    final_path = os.path.join(CKPT_DIR, "dqn_final")
    model.save(final_path)
    env.close()
    print(f"✅ Done. Saved {final_path}.zip")


def main() -> None:
    print("=== Space Invaders DQN Training ===\n")
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
        checkpoints.sort(key=lambda x: x[1])   # ascending so earliest is [1]
        chosen = prompt_checkpoint(checkpoints)
        train(mode="resume", resume_path=chosen, total_timesteps=300_000)

    elif choice == "3":
        train(mode="smoketest", total_timesteps=5_000)


if __name__ == "__main__":
    main()
