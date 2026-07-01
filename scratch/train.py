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
import zipfile
from pathlib import Path

import numpy as np
import torch

from scratch.network import QNetwork, QNetworkLegacy
from scratch.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from scratch.agent import Agent
from scratch.hyperparam import Hyperparameters
from scratch.evaluate import greedy_action
from shared.preprocessing import env_fixed


# Project root = parent of scratch/. Resolve relative paths from here so the
# script works regardless of the CWD it was launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _PROJECT_ROOT / "runs"


class _TeeWriter:
	"""File-like object that writes to both an inner stream and a log file.

	Used to tee every print(...) so the heartbeat survives Kaggle cell-output
	truncation. `inner` is what we replace on sys.stdout (e.g. the original
	stdout), `log_fh` is the open file handle for the on-disk log.

	Note: only `write` and `flush` are overridden, which is everything
	`print(...)` and `sys.stdout` need. Other file ops (e.g. `isatty`) fall
	back to the inner stream.
	"""
	def __init__(self, inner, log_fh):
		self._inner = inner
		self._log = log_fh

	def write(self, s):
		try:
			self._inner.write(s)
		except Exception:
			pass # never let logging kill the training loop
		try:
			self._log.write(s)
		except Exception:
			pass

	def flush(self):
		try:
			self._inner.flush()
		except Exception:
			pass
		try:
			self._log.flush()
		except Exception:
			pass


def write_rescue_zip(runs_dir: Path, out_path: Path, keep_recent: int = 5) -> int:
	"""Write a compact rescue zip to `out_path`.

	The zip is the on-disk safety net for Kaggle sessions: it's the one
	artifact that survives notebook re-clone (it lives at /kaggle/working/
	runs.zip, outside the repo dir that Cell 1 rmtree's). We keep only the
	N most recent step checkpoints (~20MB each) plus final.pt / crashsave.pt
	if they exist, plus the CSVs and train.log. So a 2M-step run produces a
	zip with ~5 * 20MB = 100MB of checkpoints plus a few KB of CSVs - small
	enough to be cheap, large enough to be a full resume point.

	Returns the number of files written. Errors are swallowed so a
	zip failure never aborts training.
	"""
	try:
		# Collect step checkpoints, sort by step number descending.
		step_ckpts = sorted(
			(p for p in runs_dir.glob("dqn_scratch_step_*.pt")),
			key=lambda p: int(p.stem.split("_")[-1]),
			reverse=True,
		)
		step_ckpts = step_ckpts[:keep_recent]

		# Always-include artifacts if they exist.
		always = []
		for name in ("dqn_scratch_final.pt", "dqn_scratch_crashsave.pt"):
			p = runs_dir / name
			if p.exists():
				always.append(p)
		csvs = []
		for name in ("dqn_scratch_eval.csv", "dqn_scratch_metrics.csv", "train.log"):
			p = runs_dir / name
			if p.exists():
				csvs.append(p)

		# Write to a tmp path first, then atomic-rename so a partially-written
		# zip never replaces a good one.
		tmp = out_path.with_suffix(out_path.suffix + ".tmp")
		with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
			# No compression on .pt files: they're already random-ish floats
			# so deflate barely shrinks them, and STORED is much faster.
			for p in (*step_ckpts, *always, *csvs):
				zf.write(p, arcname=f"spaceinvaderrl/runs/{p.name}")
		tmp.replace(out_path)
		return len(step_ckpts) + len(always) + len(csvs)
	except Exception as e:  # noqa: BLE001 - logging must never crash training
		print(f" [warn] rescue zip write failed: {e!r}")
		return 0


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
	p.add_argument("--device", type=str, default=None,
				 choices=["auto", "cpu", "cuda"],
				 help="Override hp.device. 'auto' picks CUDA when available, "
					 "else CPU. Default 'auto'.")
	p.add_argument("--auto-zip-freq", type=int, default=None,
				 help="Override hp.auto_zip_freq. Steps between rescue-zip writes. "
					 "0 disables auto-zip. Default 200_000.")
	p.add_argument("--auto-zip-keep", type=int, default=None,
				 help="Override hp.auto_zip_keep. Recent step checkpoints "
					 "kept in the zip. Default 5.")
	p.add_argument("--prio-alpha", type=float, default=None,
				 help="Override hp.prio_alpha. Prioritization exponent (0=uniform, 1=pure TD-error). Default 0.6.")
	p.add_argument("--prio-beta", type=float, default=None,
				 help="Override hp.prio_beta. Initial IS weight exponent. Default 0.4.")
	p.add_argument("--use-legacy-network", action="store_true",
				 help="Use QNetworkLegacy (standard DQN) instead of Dueling+Noisy.")
	p.add_argument("--use-uniform-buffer", action="store_true",
				 help="Use uniform ReplayBuffer instead of PrioritizedReplayBuffer.")
	return p.parse_args()


def set_seed(seed: int) -> None:
	"""Seed Python, numpy, and torch for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def maybe_resume(path, q_online, q_target, optimizer, device):
	"""Load a checkpoint if given. Returns (start_step, hp).

	start_step is 1 for fresh training, ckpt['step']+1 for resumed.

	`device` is the torch device (str, "cpu" or "cuda") the optimizer was
	built against. The checkpoint is loaded with map_location=device so
	its optimizer-state tensors land on the same device as the params
	they're tracking (otherwise the first optimizer.step() after resume
	raises "tensors on different devices").

	Special values for `path`:
	- None: fresh training.
	- "latest": pick the highest-step dqn_scratch_step_*.pt in _RUNS_DIR,
	  falling back to final.pt or crashsave.pt if no step ckpts exist.
	  Used by emergency-resume cells in the notebook after a Kaggle
	  disconnect, where the user wants to pick up "wherever I was" without
	  knowing the exact step number.
	- anything else: treated as a path. Relative paths resolve against
	  _PROJECT_ROOT (so `--resume runs/dqn_scratch_step_5000000.pt` works
	  from any CWD).
	"""
	if path is None:
		return 1, Hyperparameters()

	if path == "latest":
		candidates = sorted(
			_RUNS_DIR.glob("dqn_scratch_step_*.pt"),
			key=lambda p: int(p.stem.split("_")[-1]),
			reverse=True,
		)
		if candidates:
			ckpt_path = str(candidates[0])
			print(f"[resume] 'latest' -> {ckpt_path}")
		else:
			# Fall back to final.pt / crashsave.pt if no step ckpts yet.
			for fallback in ("dqn_scratch_final.pt", "dqn_scratch_crashsave.pt"):
				fb = _RUNS_DIR / fallback
				if fb.exists():
					ckpt_path = str(fb)
					print(f"[resume] 'latest' (no step ckpts) -> {ckpt_path}")
					break
			else:
				print(f"[resume] 'latest' but no checkpoints in {_RUNS_DIR}; starting fresh.")
				return 1, Hyperparameters()
	else:
		ckpt_path = path
		if not os.path.isabs(ckpt_path):
			ckpt_path = str(_PROJECT_ROOT / ckpt_path)
		if not os.path.exists(ckpt_path):
			print(f"[resume] checkpoint not found at {ckpt_path}; starting fresh.")
			return 1, Hyperparameters()

	ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
	q_online.load_state_dict(ckpt["model_state_dict"])
	q_target.load_state_dict(ckpt["model_state_dict"])
	# NOTE: optimizer_state_dict is intentionally NOT loaded. Adam's
	# exp_avg / exp_avg_sq tensors are tied to specific param objects by
	# id(); when we build a fresh optimizer against fresh params (even if
	# we copy the weights over), the param ids don't match, so loading
	# state raises a "tensors on different devices" error in modern PyTorch.
	# We pay a small cost: the first ~5k steps of training rebuild Adam's
	# momentum from scratch. The model weights and target net are preserved,
	# so the agent's *policy* resumes correctly -- this is the part that
	# matters for evaluation. (Mnih 2015 uses a learning_starts=50k warmup
	# for exactly this reason; our 5k is just enough for momentum to ramp.)
	# if "optimizer_state_dict" in ckpt:
	# 	optimizer.load_state_dict(ckpt["optimizer_state_dict"])
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

	# Tee stdout so every print() also lands on disk. Kaggle's cell output
	# truncates beyond ~20MB, so this is the durable copy. Resume-safety:
	# we append (don't clobber) and prepend a session header so a fresh
	# run is easy to spot in the file.
	log_path = _RUNS_DIR / "train.log"
	_log_fh = open(log_path, "a", encoding="utf-8", buffering=1) # line-buffered
	_orig_stdout = sys.stdout
	sys.stdout = _TeeWriter(_orig_stdout, _log_fh)
	# Header makes the file grep-friendly.
	print(f"\n{'=' * 60}")
	print(f"[session] start_time = {time.strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"[session] pid = {os.getpid()}")
	print(f"[session] resume = {args.resume or '(none, fresh run)'}")
	print(f"{'=' * 60}")

	# Per-heartbeat metrics CSV (loss + Q stats). Lives next to eval.csv.
	# Header is written on first run; resume appends (file already exists).
	metrics_csv = _RUNS_DIR / "dqn_scratch_metrics.csv"
	metrics_csv_exists = metrics_csv.exists()

	num_actions = 6
	if args.use_legacy_network:
		q_online = QNetworkLegacy(num_actions=num_actions)
		q_target = QNetworkLegacy(num_actions=num_actions)
		print("[setup] Using QNetworkLegacy (standard DQN, no dueling/noisy)")
	else:
		q_online = QNetwork(num_actions=num_actions)
		q_target = QNetwork(num_actions=num_actions)
		print("[setup] Using QNetwork (Dueling + Noisy Nets)")
	q_target.eval() # documents intent; NatureCNN has no dropout/BN

	# Resolve device BEFORE moving networks. The optimizer holds per-
	# parameter state (Adam's exp_avg, exp_avg_sq) that lives on the
	# same device as the parameters it tracks. Order matters:
	#   1. Build networks on CPU (default).
	#   2. Resolve target device ("cuda" if available, else "cpu").
	#   3. Move networks to device.
	#   4. Build optimizer against the now-device-resident parameters.
	#   5. Load checkpoint with map_location=device so its optimizer
	#      state tensors land on the same device as the params they
	#      describe.
	# The old order (build optimizer on CPU, then .to(cuda), then load
	# checkpoint with map_location=cpu) hit a 7M-step resume with:
	#   RuntimeError: Expected all tensors to be on the same device,
	#   but found at least two devices, cuda:0 and cpu!
	# at the first optimizer.step() after resume, because Adam's
	# exp_avg/exp_avg_sq were loaded onto CPU (map_location=cpu) but
	# the params they'd been tracking were now on CUDA.
	#
	# Read the device hint from args (CLI) directly: hp is bound later
	# by maybe_resume(), and using a placeholder would silently ignore
	# any --device override on this run.
	requested_device = args.device if args.device is not None else "auto"
	cuda_ok = torch.cuda.is_available()
	if requested_device == "auto":
		device = "cuda" if cuda_ok else "cpu"
	elif requested_device == "cuda":
		if cuda_ok:
			device = "cuda"
		else:
			print(f"[setup] WARNING: --device cuda requested but "
				 f"torch.cuda.is_available() is False; falling back to CPU.")
			print(f"[setup] (Check: right torch build? GPU enabled in kernel? "
				 f"`nvidia-smi` works?)")
			device = "cpu"
	else:
		device = "cpu"
	# Move networks FIRST so the optimizer tracks CUDA-resident params
	# (or CPU-resident if device=="cpu") from its very first .step().
	q_online.to(device)
	q_target.to(device)
	optimizer = torch.optim.Adam(q_online.parameters(), lr=Hyperparameters.lr)

	start_step, hp = maybe_resume(args.resume, q_online, q_target, optimizer, device)

	if args.total_steps is not None:
		hp.total_steps = args.total_steps
	if args.save_freq is not None:
		hp.save_freq = args.save_freq
	if args.eval_freq is not None:
		hp.eval_freq = args.eval_freq
	if args.seed is not None:
		hp.seed = args.seed
	if args.device is not None:
		hp.device = args.device
	if args.auto_zip_freq is not None:
		hp.auto_zip_freq = args.auto_zip_freq
	if args.auto_zip_keep is not None:
		hp.auto_zip_keep = args.auto_zip_keep
	if args.prio_alpha is not None:
		hp.prio_alpha = args.prio_alpha
	if args.prio_beta is not None:
		hp.prio_beta = args.prio_beta
	if args.use_legacy_network:
		hp.use_legacy_network = True
	if args.use_uniform_buffer:
		hp.use_prioritized = False

	set_seed(hp.seed)
	print(f"[setup] device = {device} (requested={requested_device}, cuda_available={cuda_ok})")
	if device == "cuda":
		print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")
	print(f"[setup] total_steps = {hp.total_steps:,}")
	print(f"[setup] save_freq = {hp.save_freq:,}, eval_freq = {hp.eval_freq:,}")
	print(f"[setup] train_freq = 4 (hardcoded to match paper)")
	print(f"[setup] min_repeat = {hp.min_repeat} (MinActionRepeat wrapper applied via env_fixed)")
	print(f"[setup] auto_zip_freq = {hp.auto_zip_freq:,} (rescue zip every N steps, 0=off)")
	if hp.use_prioritized:
		print(f"[setup] prio_alpha = {hp.prio_alpha}, beta = {hp.prio_beta}→{hp.prio_beta_end} "
			  f"(IS correction ramps over first {hp.prio_beta_frac:.0%} of training)")

	env = env_fixed(seed=hp.seed, render_mode=None, min_repeat=hp.min_repeat)
	agent = Agent()
	if hp.use_prioritized:
		buffer = PrioritizedReplayBuffer(
			capacity=hp.buffer_size,
			obs_shape=(4, 84, 84),
			alpha=hp.prio_alpha,
			beta=hp.prio_beta,
			beta_end=hp.prio_beta_end,
			beta_frac=hp.prio_beta_frac,
		)
		print(f"[setup] Using PrioritizedReplayBuffer "
			  f"(alpha={hp.prio_alpha}, beta={hp.prio_beta}→{hp.prio_beta_end})")
	else:
		buffer = ReplayBuffer(hp.buffer_size, obs_shape=(4, 84, 84))
		print("[setup] Using uniform ReplayBuffer")

	if not eval_csv_exists:
		with open(eval_csv, "w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["step", "mean_reward", "std_reward", "wall_time_s"])

	if not metrics_csv_exists:
		with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow([
				"step", "wall_time_s",
				"loss", "mean_q", "max_q", "min_q",
				"eps", "last_ep_R", "mean_ep_R", "episodes",
				"buffer_size", "steps_per_s",
			])

	obs, _ = env.reset(seed=hp.seed)
	episode_reward = 0.0
	episode_count = 0
	last_print = time.time()
	start_time = last_print
	# Rolling-mean reward over the last N episodes. One episode's reward is
	# noise; an average of the last 10 is the actual signal. `collections.deque`
	# with maxlen gives O(1) append + auto-eviction.
	from collections import deque
	ep_recent = deque(maxlen=10)
	# Most recent training metrics (loss, Q stats) for the heartbeat. Initialized
	# to None so the first heartbeat (before any train_step) prints "n/a" instead
	# of crashing on an unbound variable.
	last_metrics = None
	t = start_step - 1 # so the except blocks can read t after a crash

	print("\n[training] starting main loop...")
	try:
		for t in range(start_step, hp.total_steps + 1):
			if args.use_legacy_network:
				# ε-greedy for legacy QNetwork
				eps = agent.epsilon_at(t, hp)
				action = agent.select_action_eps(obs, q_online, eps, num_actions)
			else:
				# Noisy Nets handle exploration — no epsilon needed
				action = agent.select_action(obs, q_online)
			next_obs, reward, terminated, truncated, _ = env.step(action)
			# IMPORTANT: only mark done on natural termination. Truncation
			# (time-limit) still allows the bootstrap term to apply.
			done = bool(terminated)
			buffer.push(obs, action, reward, next_obs, done)
			episode_reward += float(reward)
			obs = next_obs

			if terminated or truncated:
				episode_count += 1
				ep_recent.append(float(episode_reward))
				obs, _ = env.reset()
				# Heartbeat every ~10s so Kaggle doesn't idle-out.
				now = time.time()
				if now - last_print > 10.0:
					elapsed = now - start_time
					sps = (t - start_step + 1) / max(elapsed, 1e-6)
					mean_ep_r = (sum(ep_recent) / len(ep_recent)) if ep_recent else 0.0
					if last_metrics is not None:
						loss_s = f"{last_metrics['loss']:.4f}"
						q_s = (f"q[{last_metrics['min_q']:+.2f},"
							 f" {last_metrics['mean_q']:+.2f},"
							 f" {last_metrics['max_q']:+.2f}]")
					else:
						loss_s = "n/a"
						q_s = "q[n/a, n/a, n/a]"
					print(f" step {t:>7,}/{hp.total_steps:,} | "
						 f"eps {eps:.3f} | "
						 f"loss {loss_s:>7s} | "
						 f"{q_s} | "
						 f"ep {episode_count:>5d} | "
						 f"R_last {episode_reward:6.1f} | "
						 f"R_mean{len(ep_recent):>2d} {mean_ep_r:6.1f} | "
						 f"buf {buffer.size:>6,} | "
						 f"{sps:5.1f} sps")
					# Append to the metrics CSV so we can plot loss/Q curves
					# post-hoc. Failures here must not abort training.
					try:
						with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
							writer = csv.writer(f)
							writer.writerow([
								t,
								f"{elapsed:.1f}",
								"" if last_metrics is None else f"{last_metrics['loss']:.6f}",
								"" if last_metrics is None else f"{last_metrics['mean_q']:.4f}",
								"" if last_metrics is None else f"{last_metrics['max_q']:.4f}",
								"" if last_metrics is None else f"{last_metrics['min_q']:.4f}",
								f"{eps:.4f}",
								f"{episode_reward:.2f}",
								f"{mean_ep_r:.2f}",
								episode_count,
								buffer.size,
								f"{sps:.2f}",
							])
					except Exception as csv_err: # noqa: BLE001 - logging must never crash training
						print(f" [warn] metrics CSV append failed: {csv_err!r}")
					last_print = now
				episode_reward = 0.0

			# Learn once we have enough samples and every train_freq steps.
			if t >= hp.learning_starts and (t % 4 == 0):
				last_metrics = agent.train_step(q_online, q_target, optimizer, buffer, hp, step=t)
				agent.post_train_step(buffer, last_metrics)

			# Refresh target net.
			if t % hp.target_update == 0:
				q_target.load_state_dict(q_online.state_dict())

			# Periodic checkpoint.
			if t % hp.save_freq == 0:
				ckpt_path = _RUNS_DIR / f"dqn_scratch_step_{t}.pt"
				save_checkpoint(ckpt_path, t, q_online, optimizer, hp)
				print(f" [ckpt] saved {ckpt_path.name}")

			# Periodic rescue zip. Writes /kaggle/working/runs.zip with the
			# N most recent step checkpoints + final/crashsave + CSVs + log.
			# Lives OUTSIDE the repo dir so it survives Cell 1 re-clone.
			# Default off (0); enable with hp.auto_zip_freq > 0.
			if hp.auto_zip_freq > 0 and t % hp.auto_zip_freq == 0:
				rescue_zip = _PROJECT_ROOT.parent / "runs.zip"
				n_files = write_rescue_zip(_RUNS_DIR, rescue_zip,
										   keep_recent=hp.auto_zip_keep)
				if n_files > 0:
					print(f" [rescue] wrote {rescue_zip.name} "
						 f"({n_files} files, "
						 f"{rescue_zip.stat().st_size / 1e6:.1f} MB)")

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
		# Refresh the rescue zip so the on-disk backup matches the just-saved
		# crashsave state. Critical for "I stopped the cell, lost my work"
		# scenarios like the one that wiped a 1M-step run last time.
		if hp.auto_zip_freq > 0:
			rescue_zip = _PROJECT_ROOT.parent / "runs.zip"
			n_files = write_rescue_zip(_RUNS_DIR, rescue_zip,
									   keep_recent=hp.auto_zip_keep)
			if n_files > 0:
				print(f"[interrupt] refreshed {rescue_zip.name} "
					 f"({rescue_zip.stat().st_size / 1e6:.1f} MB)")
	finally:
		env.close()
		# Flush + close the tee'd log handle and restore stdout. Wrapped in
		# try/except so a logging error at shutdown never masks the real
		# exit reason (e.g. MemoryError).
		try:
			sys.stdout = _orig_stdout
		except Exception:
			pass
		try:
			_log_fh.flush()
			_log_fh.close()
		except Exception:
			pass

	final_path = _RUNS_DIR / "dqn_scratch_final.pt"
	save_checkpoint(final_path, hp.total_steps, q_online, optimizer, hp)
	total_time = time.time() - start_time
	print(f"\n[done] saved {final_path.name}")
	print(f"[done] total wall time: {total_time/60:.1f} min "
		 f"({(hp.total_steps - start_step + 1)/max(total_time,1e-6):.1f} steps/s)")
	# Final rescue zip at the natural end of training.
	if hp.auto_zip_freq > 0:
		rescue_zip = _PROJECT_ROOT.parent / "runs.zip"
		n_files = write_rescue_zip(_RUNS_DIR, rescue_zip,
								   keep_recent=hp.auto_zip_keep)
		if n_files > 0:
			print(f"[done] wrote {rescue_zip.name} "
				 f"({rescue_zip.stat().st_size / 1e6:.1f} MB)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
