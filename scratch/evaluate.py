"""Greedy evaluation of a from-scratch DQN checkpoint.

Runs the agent for N episodes with eps=0 (pure greedy / argmax) and reports
mean +/- std episode reward. This is the script you use to get a number
that's comparable to `python evaluate.py` (the SB3 evaluator), but using
your own QNetwork and Agent instead of Stable Baselines3.

Why a separate file: the root evaluate.py uses SB3's DQN.load + evaluate_policy,
which only understands SB3's .zip format. This script loads a torch .pt
checkpoint and replays episodes using dqn.network.QNetwork and
dqn.agent.Agent.select_action.

Note: env construction here uses env_fixed (with min_repeat pulled from
the saved hp dict) so the eval environment exactly matches what the agent
trained on. Mismatched wrappers silently inflate or deflate scores.

Usage:
 python -m dqn.evaluate --checkpoint runs/dqn_scratch_final.pt --episodes 20
 python -m dqn.evaluate --checkpoint runs/dqn_scratch_final.pt --episodes 50
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from scratch.network import QNetwork
from scratch.hyperparam import Hyperparameters
from shared.preprocessing import env_fixed


# Resolve relative paths from the project root (parent of scratch/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Evaluate a from-scratch DQN agent.")
	p.add_argument("--checkpoint", type=str, required=True,
				 help="Path to the .pt checkpoint (e.g. runs/dqn_scratch_final.pt).")
	p.add_argument("--episodes", type=int, default=20,
				 help="Number of episodes to roll out (default 20).")
	p.add_argument("--seed", type=int, default=12345,
				 help="Base env seed. Episode ep uses seed + ep for variety.")
	p.add_argument("--device", type=str, default="cpu",
				 help="Torch device for inference (default cpu for determinism).")
	return p.parse_args()


def load_checkpoint(path, device):
	"""Load a torch checkpoint, rebuild the QNetwork, return (net, ckpt, hp)."""
	ckpt = torch.load(path, map_location=device, weights_only=False)
	hp_dict = ckpt.get("hp", None)
	if hp_dict is None:
		hp = Hyperparameters()
	else:
		hp = Hyperparameters(**hp_dict)
	num_actions = int(getattr(hp, "num_actions", 6))
	q_net = QNetwork(num_actions=num_actions)
	q_net.load_state_dict(ckpt["model_state_dict"])
	q_net.to(device)
	q_net.eval()
	return q_net, ckpt, hp


def greedy_action(q_net, state, device):
	"""argmax over Q-values for a single observation. No epsilon, no_grad."""
	with torch.no_grad():
		s = torch.as_tensor(np.array(state, dtype=np.float32), device=device).unsqueeze(0)
		q_values = q_net(s) # (1, |A|)
	return int(q_values.argmax(dim=1).item())


def rollout_episode(env, q_net, seed, device):
	"""Play one full episode under the greedy policy. Return total reward."""
	obs, _ = env.reset(seed=seed)
	total = 0.0
	while True:
		action = greedy_action(q_net, obs, device)
		obs, reward, terminated, truncated, _ = env.step(action)
		total += float(reward)
		if terminated or truncated:
			return total


def main():
	args = parse_args()

	ckpt_path = args.checkpoint
	if not os.path.isabs(ckpt_path):
		ckpt_path = str(_PROJECT_ROOT / ckpt_path)
	if not os.path.exists(ckpt_path):
		print(f"ERROR: checkpoint not found: {ckpt_path}")
		return 1

	print("=== From-Scratch DQN Evaluation ===")
	print(f" Checkpoint : {ckpt_path}")
	print(f" Episodes : {args.episodes}")
	print(f" Seed : {args.seed}")
	print(f" Device : {args.device}")

	q_net, ckpt, hp = load_checkpoint(ckpt_path, args.device)
	step = ckpt.get("step", -1)
	min_repeat = int(getattr(hp, "min_repeat", 4))
	print(f" Loaded QNetwork (step={step:,}).")
	print(f" min_repeat = {min_repeat} (MinActionRepeat wrapper applied via env_fixed)")
	print()

	# CRITICAL: env must match training exactly. We read min_repeat from the
	# saved hp dict so a checkpoint trained with min_repeat=4 isn't silently
	# evaluated at min_repeat=4 default.
	env = env_fixed(seed=args.seed, render_mode=None, min_repeat=min_repeat)

	print(f"Running {args.episodes} greedy episodes...")
	rewards = np.zeros(args.episodes, dtype=np.float64)
	for ep in range(args.episodes):
		ep_seed = args.seed + ep
		rewards[ep] = rollout_episode(env, q_net, ep_seed, args.device)
		print(f" ep {ep+1:>3d}/{args.episodes}: reward = {rewards[ep]:7.1f}")

	env.close()

	mean = float(rewards.mean())
	std = float(rewards.std())
	print()
	print("=" * 50)
	print(f" Mean reward: {mean:8.1f} +/- {std:.1f}")
	print("=" * 50)
	print()
	print("Compare against the SB3 baseline (~304.2 +/- 123.5) to see if the")
	print("from-scratch implementation matches Stable Baselines3's DQN.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
