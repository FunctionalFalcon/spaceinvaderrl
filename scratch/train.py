"""From-scratch DQN training on Space Invaders.

Rebuilds the Mnih 2015 DQN algorithm in PyTorch without Stable Baselines3.
Trains for hp.total_steps environment steps, with periodic checkpointing,
greedy eval, and crash-safe resume.

Usage:
 python -m dqn.train # full run with defaults
 python -m dqn.train --total_steps 5000 --save_freq 5000 # smoke test
 python -m dqn.train --resume runs/dqn_scratch_resume.pt # resume

Outputs:
 runs/dqn_scratch_step_<N>.pt periodic checkpoints (full state)
 runs/dqn_scratch_final.pt saved at end of training
 runs/dqn_scratch_eval.csv (step, mean_reward, std_reward) per eval
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scratch.network import QNetwork
from scratch.replay_buffer import ReplayBuffer
from scratch.agent import Agent
from scratch.hyperparam import Hyperparameters
from scratch.evaluate import greedy_action
from shared.preprocessing import env_fixed


# Project root = parent of scratch/. Resolve relative paths from here so the
# script works regardless of the CWD it was launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train DQN from scratch on Space Invaders.")
	p.add_argument("--total_steps", type=int, default=None,
				 help="Override hp.total_steps (useful for smoke tests).")
	p.add_argument("--save_freq", type=int, default=None,
				 help="Override hp.save_freq.")
	p.add_argument("--eval_freq", type=int, default=None,
				 help="Override hp.eval_freq.")
	p.add_argument("--resume", type=str, default=None,
				 help="Path to a .pt checkpoint to resume from.")
	p.add_argument("--seed", type=int, default=None,
				 help="Override hp.seed.")
	return p.parse_args()


def set_seed(seed: int) -> None:
	"""Seed Python, numpy, and torch for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def maybe_resume(path, q_online, q_target, optimizer):
	"""Load a checkpoint if given. Returns (start_step, hp).

	start_step is 1 for fresh training, ckpt['step']+1 for resumed."""
	if path is None:
		return 1, Hyperparameters()
	ckpt_path = path
	if not os.path.isabs(ckpt_path):
		ckpt_path = str(_PROJECT_ROOT / ckpt_path)
	if not os.path.exists(ckpt_path):
		print(f"[resume] checkpoint not found at {ckpt_path}; starting fresh.")
		return 1, Hyperparameters()

	ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
	q_online.load_state_dict(ckpt["model_state_dict"])
	q_target.load_state_dict(ckpt["model_state_dict"])
	if "optimizer_state_dict" in ckpt:
		optimizer.load_state_dict(ckpt["optimizer_state_dict"])
	hp = Hyperparameters(**ckpt["hp"]) if "hp" in ckpt else Hyperparameters()
	start_step = int(ckpt.get("step", 0)) + 1
	print(f"[resume] loaded {ckpt_path} at step {start_step:,}")
	return start_step, hp


def save_checkpoint(path, step, q_online, optimizer, hp):
	"""Write to a .tmp file first, then rename. Atomic on most filesystems."""
	tmp = path.with_suffix(path.suffix + ".tmp")
	torch.save({
		"step": step,
		"model_state_dict": q_online.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"hp": hp.__dict__,
	}, tmp)
	tmp.replace(path)


def run_greedy_eval(env, q_net, n_episodes, seed_base, device):
	"""Roll out n_episodes greedy episodes. Returns (mean, std)."""
	rewards = np.zeros(n_episodes, dtype=np.float64)
	for ep in range(n_episodes):
		obs, _ = env.reset(seed=seed_base + ep)
		total = 0.0
		while True:
			a = greedy_action(q_net, obs, device)
			obs, r, term, trunc, _ = env.step(a)
			total += float(r)
			if term or trunc:
				break
		rewards[ep] = total
	return float(rewards.mean()), float(rewards.std())


def main() -> int:
	args = parse_args()

	os.makedirs(_RUNS_DIR, exist_ok=True)
	eval_csv = _RUNS_DIR / "dqn_scratch_eval.csv"
	eval_csv_exists = eval_csv.exists()

	num_actions = 6
	q_online = QNetwork(num_actions=num_actions)
	q_target = QNetwork(num_actions=num_actions)
	q_target.eval() # documents intent; NatureCNN has no dropout/BN
	optimizer = torch.optim.Adam(q_online.parameters(), lr=Hyperparameters.lr)

	start_step, hp = maybe_resume(args.resume, q_online, q_target, optimizer)

	if args.total_steps is not None:
		hp.total_steps = args.total_steps
	if args.save_freq is not None:
		hp.save_freq = args.save_freq
	if args.eval_freq is not None:
		hp.eval_freq = args.eval_freq
	if args.seed is not None:
		hp.seed = args.seed

	set_seed(hp.seed)
	device = hp.device if (hp.device == "cpu" or torch.cuda.is_available()) else "cpu"
	q_online.to(device)
	q_target.to(device)
	print(f"[setup] device = {device}")
	print(f"[setup] total_steps = {hp.total_steps:,}")
	print(f"[setup] save_freq = {hp.save_freq:,}, eval_freq = {hp.eval_freq:,}")
	print(f"[setup] train_freq = 4 (hardcoded to match paper)")
	print(f"[setup] min_repeat = {hp.min_repeat} (MinActionRepeat wrapper applied via env_fixed)")

	env = env_fixed(seed=hp.seed, render_mode=None, min_repeat=hp.min_repeat)
	agent = Agent()
	buffer = ReplayBuffer(hp.buffer_size, obs_shape=(4, 84, 84))

	if not eval_csv_exists:
		with open(eval_csv, "w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["step", "mean_reward", "std_reward", "wall_time_s"])

	obs, _ = env.reset(seed=hp.seed)
	episode_reward = 0.0
	episode_count = 0
	last_print = time.time()
	start_time = last_print
	t = start_step - 1 # so the except blocks can read t after a crash

	print("\n[training] starting main loop...")
	try:
		for t in range(start_step, hp.total_steps + 1):
			eps = agent.epsilon_at(t, hp)
			action = agent.select_action(obs, q_online, eps, num_actions)
			next_obs, reward, terminated, truncated, _ = env.step(action)
			# IMPORTANT: only mark done on natural termination. Truncation
			# (time-limit) still allows the bootstrap term to apply.
			done = bool(terminated)
			buffer.push(obs, action, reward, next_obs, done)
			episode_reward += float(reward)
			obs = next_obs

			if terminated or truncated:
				episode_count += 1
				obs, _ = env.reset()
				# Heartbeat every ~10s so Kaggle doesn't idle-out.
				now = time.time()
				if now - last_print > 10.0:
					elapsed = now - start_time
					sps = (t - start_step + 1) / max(elapsed, 1e-6)
					print(f" step {t:>7,}/{hp.total_steps:,} | "
						 f"eps {eps:.3f} | "
						 f"episodes {episode_count} | "
						 f"last_ep_R {episode_reward:6.1f} | "
						 f"buffer {buffer.size:,} | "
						 f"{sps:.1f} steps/s")
					last_print = now
				episode_reward = 0.0

			# Learn once we have enough samples and every train_freq steps.
			if t >= hp.learning_starts and (t % 4 == 0):
				loss = agent.train_step(q_online, q_target, optimizer, buffer, hp)

			# Refresh target net.
			if t % hp.target_update == 0:
				q_target.load_state_dict(q_online.state_dict())

			# Periodic checkpoint.
			if t % hp.save_freq == 0:
				ckpt_path = _RUNS_DIR / f"dqn_scratch_step_{t}.pt"
				save_checkpoint(ckpt_path, t, q_online, optimizer, hp)
				print(f" [ckpt] saved {ckpt_path.name}")

			# Periodic greedy eval (and CSV append).
			if t % hp.eval_freq == 0:
				mean_r, std_r = run_greedy_eval(env, q_online, hp.eval_episodes,
												hp.seed + 10_000, device)
				with open(eval_csv, "a", newline="", encoding="utf-8") as f:
					writer = csv.writer(f)
					writer.writerow([t, f"{mean_r:.3f}", f"{std_r:.3f}",
									 f"{time.time() - start_time:.1f}"])
				print(f" [eval] step {t:,}: mean_R = {mean_r:7.1f} +/- {std_r:.1f}")

	except (MemoryError, RuntimeError) as e:
		crash_path = _RUNS_DIR / "dqn_scratch_crashsave.pt"
		save_checkpoint(crash_path, t, q_online, optimizer, hp)
		print(f"\n[crash] {type(e).__name__}: {e}")
		print(f"[crash] saved {crash_path}; reraise.")
		raise
	except KeyboardInterrupt:
		crash_path = _RUNS_DIR / "dqn_scratch_crashsave.pt"
		save_checkpoint(crash_path, t, q_online, optimizer, hp)
		print(f"\n[interrupt] saved {crash_path}. "
			 f"Resume with --resume {crash_path.name}")
	finally:
		env.close()

	final_path = _RUNS_DIR / "dqn_scratch_final.pt"
	save_checkpoint(final_path, hp.total_steps, q_online, optimizer, hp)
	total_time = time.time() - start_time
	print(f"\n[done] saved {final_path.name}")
	print(f"[done] total wall time: {total_time/60:.1f} min "
		 f"({(hp.total_steps - start_step + 1)/max(total_time,1e-6):.1f} steps/s)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
