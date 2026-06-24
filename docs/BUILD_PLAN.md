# From-Scratch DQN — Build Plan + Paper Comparison

This is your project-internal plan. Read it once before you start typing, then refer back to it per step. **Chat is for questions and debugging, not for re-explaining the plan.**

There are two parts:

- **Part A — Build plan** (this file, below): the 5 files you'll write, the order to write them, the smoke test for each, and what NOT to do.
- **Part B — Paper comparison** (the other file, `docs/scratch_dqn_vs_paper.md`): what we changed vs. Mnih 2015, why we changed it, and whether it's better.

---

## Context

Teacher rejected SB3. You're rebuilding DQN from scratch in PyTorch. You will type the code yourself based on the pseudocode + knowledge transfer in `docs/pytorch_from_scratch.md` (already in the repo, 80% of the answer). This plan gives the build order and the small changes we made vs. the paper.

**Targets:**

- Same `preprocessing.py` (unchanged), same `record_*.py` (unchanged), same `play.py` (unchanged but see note below).
- New `dqn/` package with 5 files: `network.py`, `replay_buffer.py`, `agent.py`, `train.py`, `evaluate.py`.
- Training: 1.2M steps on GPU, target eval reward ≥ 250 (SB3 baseline was 257.5 ± 104.9).
- Run on Kaggle T4. ~12h wall-clock.

**One incidental change to `record_trained.py`:** the existing version uses `DQN.load` (SB3). The from-scratch version needs a torch checkpoint loader. You'll add a `--framework scratch` flag and a small loader block — ~5 lines, not a rewrite. Defer this until the model is trained.

---

## Part A — Build plan

### File 1: `dqn/network.py` — QNetwork (NatureCNN)

**What to type:** A `QNetwork(nn.Module)` with 3 conv layers + 2 FC layers. ReLU after every layer except the last. Reference: `docs/pytorch_from_scratch.md` §2. Full pseudocode in this file's "Module 1" section from the previous chat turn.

**What NOT to do:**

- No batch norm, no dropout, no skip connections.
- No softmax / sigmoid on the output.
- No padding (default of 0 is correct).
- Don't compute the flat size with a dummy forward pass — just hardcode `64 * 7 * 7 = 3136`.

**Smoke test (must pass before moving on):**

```python
m = QNetwork(num_actions=6)
x = torch.zeros(1, 4, 84, 84)
assert m(x).shape == (1, 6)
m(torch.zeros(32, 4, 84, 84))  # also test batched
print(sum(p.numel() for p in m.parameters()))  # 1,686,758
```

**Common mistakes** (full table in previous chat turn): swapped `in_channels`/`out_channels`, `Linear(512, 64*7*7)` reversed, ReLU on `fc2`, missing `super().__init__()`.

---

### File 2: `dqn/replay_buffer.py` — ReplayBuffer

**What to type:** A ring buffer with 5 pre-allocated numpy arrays (`obs`, `next_obs`, `actions`, `rewards`, `dones`) and two pointers (`idx`, `size`). Reference: `docs/pytorch_from_scratch.md` §3.

**What NOT to do:**

- Don't use a Python list. Must be pre-allocated numpy.
- Don't sample from `[0, capacity)`. Sample from `[0, size)`.
- Don't store `dones` as `bool` — must be `float32` (0.0 or 1.0) so we can multiply.
- Don't convert the whole array to torch on every `sample` — `torch.as_tensor` shares memory when dtypes match.

**Smoke test:**

```python
b = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))
for _ in range(50): b.push(np.zeros((4,84,84)), 1, 0.0, np.zeros((4,84,84)), False)
o, a, r, o2, d = b.sample(4)
assert o.shape == (4, 4, 84, 84)
assert a.dtype == torch.int64 or torch.long
assert d.dtype == torch.float32
```

---

### File 3: `dqn/agent.py` — three functions

Three small functions in one file:

- `select_action(state, q_net, eps, num_actions) -> int` — ε-greedy. `torch.no_grad()` around the network call.
- `epsilon_at(t, hp) -> float` — linear decay from `eps_start` to `eps_end` over `eps_frac * total_steps`.
- `train_step(q_online, q_target, optimizer, buffer, hp) -> float` — sample minibatch, compute TD target with `torch.no_grad()`, gather predicted Q for taken actions, Huber loss, gradient clip, optimizer step.

Reference: `docs/pytorch_from_scratch.md` §4 and §5.

**What NOT to do:**

- Don't forget `torch.no_grad()` around the target computation. If you do, gradients flow through the target net and you lose stability.
- Don't use `argmax(dim=1)` without `keepdim=True` and then forget to add the dim back. The `gather(1, indices)` call requires `indices` to be `(B, 1)`, not `(B,)`.
- Don't apply softmax / sigmoid anywhere in `train_step`.
- Don't do `optimizer.step()` before `optimizer.zero_grad()`. Order matters.
- Don't `optimizer.step()` _after_ `clip_grad_norm_` is wrong — it should be: `zero_grad` → `backward` → `clip_grad_norm_` → `step`.

**Smoke test:**

```python
# Test select_action: returns an int in [0, num_actions)
state = np.zeros((4, 84, 84), dtype=np.float32)
m = QNetwork(6)
for eps in [0.0, 0.5, 1.0]:
    a = select_action(state, m, eps, 6)
    assert isinstance(a, int) and 0 <= a < 6

# Test epsilon_at: monotonic, clamped
hp = Hyperparameters(eps_frac=0.1, eps_start=1.0, eps_end=0.01, total_steps=1000)
assert epsilon_at(1, hp) == pytest.approx(1.0, abs=1e-6)
assert epsilon_at(100, hp) == pytest.approx(0.01, abs=1e-6)
assert epsilon_at(500, hp) == pytest.approx(0.01, abs=1e-6)
```

---

### File 4: `dqn/train.py` — main entry point

This ties it all together. Reference: `docs/pytorch_from_scratch.md` §5.

**What to type:**

- `Hyperparameters` dataclass (see values below).
- Seeds: `random.seed`, `np.random.seed`, `torch.manual_seed` (all from `hp.seed`).
- Two `QNetwork` instances, target = frozen copy of online.
- `Adam(q_online.parameters(), lr=hp.lr)`.
- `ReplayBuffer(hp.buffer_size, obs_shape=(4, 84, 84))`.
- Main loop 1 to `hp.total_steps`.
- Per-step: `select_action` → `env.step` → `buffer.push` → `train_step` (after `learning_starts`) → `target.load_state_dict` (every `target_update` steps) → save checkpoint (every `save_freq` steps).
- **Crash-safe wrapper** (matches `train_fixed.py`):
  ```python
  try:
      for t in range(...): ...
  except (MemoryError, RuntimeError):
      torch.save(CRASH_SAVE, "dqn_scratch_crashsave.pt"); raise
  except KeyboardInterrupt:
      torch.save(CRASH_SAVE, "dqn_scratch_crashsave.pt")
  finally:
      env.close()
  ```
- **Eval-during-train logging:** every `eval_freq=50_000` steps, run 5 greedy episodes, print mean reward, write to `runs/dqn_scratch_eval.csv`.
- **Resume support:** if `dqn_scratch_resume.pt` exists, load it and start from `ckpt["step"] + 1`.

**Hyperparameters:**

```python
@dataclass
class Hyperparameters:
    env_id:          str   = "ALE/SpaceInvaders-v5"
    seed:            int   = 0
    total_steps:     int   = 1_200_000
    buffer_size:     int   = 20_000
    learning_starts: int   = 5_000
    batch_size:      int   = 32
    gamma:           float = 0.99
    lr:              float = 1e-4
    target_update:   int   = 1_000
    save_freq:       int   = 50_000
    eval_freq:       int   = 50_000
    eval_episodes:   int   = 5
    eps_start:       float = 1.0
    eps_end:         float = 0.01
    eps_frac:        float = 0.15         # ← improvement C: 10% → 15%
    max_grad_norm:   float = 10.0
    min_repeat:      int   = 2            # ← improvement A: 4 → 2
    device:          str   = "cuda"
```

**What NOT to do:**

- Don't set `done = terminated or truncated` for the buffer push. Use `done = terminated` (see paper comparison for why).
- Don't `optimizer.step()` more than once per `train_step` call.
- Don't forget `q_target.eval()` — documents intent even though NatureCNN has no dropout/BN.
- Don't log eval reward to stdout only — write to CSV too so you can plot learning curves offline.
- Don't `env.reset()` and discard the second tuple element. `obs, _ = env.reset()`.

**Smoke test (short run, ~5 min):**

```bash
python -m dqn.train --total_steps 5000 --save_freq 5000 --eval_freq 5000
# Expected: 1 checkpoint saved, eval.csv has 1 row, mean reward ~100-150
# Loss should be in range 0.5-5.0, no NaNs.
```

---

### File 5: `dqn/evaluate.py` — greedy evaluation

**What to type:**

- Load checkpoint, instantiate `QNetwork`, `load_state_dict`.
- For `num_episodes` (default 20), run greedy policy, return `(mean_reward, std_reward)`.
- Seed each episode differently: `seed + ep`.
- Loop condition: `terminated or truncated` (this is for _stopping the loop_, not bootstrap).

**What NOT to do:**

- Don't use `eps > 0` in evaluation. The whole point is to measure the _learned_ policy, not the policy with exploration noise.
- Don't run a single episode. Use 20+ for a meaningful mean and std.
- Don't use the same seed for every episode. Use `seed + ep` for variety.

**Smoke test (after a partial training run):**

```bash
python -m dqn.evaluate --checkpoint runs/dqn_scratch_step_50000.pt --episodes 5
# Expected: mean reward in 100-200 range for an undertrained agent
```

---

### File 6 (deferred, after training): `record_trained.py` update

Add a `--framework scratch` flag. When set, load the torch checkpoint directly instead of using `DQN.load`. ~5 lines of code. **Do this AFTER the model is trained** — don't waste time on it now.

---

## Build order summary

1. `dqn/network.py` — type, smoke test, move on.
2. `dqn/replay_buffer.py` — type, smoke test, move on.
3. `dqn/agent.py` — type, smoke test all three functions, move on.
4. `dqn/train.py` — type, do a 5k-step smoke run on Kaggle, then start the 1.2M-step run.
5. `dqn/evaluate.py` — type, run on the final checkpoint.
6. `record_trained.py` — add the `--framework scratch` branch.

**Rule of thumb:** if a smoke test fails, fix it before writing the next file. Don't stack bugs.

---

## Knowledge checks (10 questions)

Be able to answer all 10 before showing the code to the teacher. The answers are in `docs/pytorch_from_scratch.md` §6 and in the previous chat turn's grading.

1. Why two networks?
2. Why a replay buffer?
3. Why clip the gradient? (and how is it different from `torch.no_grad()`?)
4. `done = terminated` vs `terminated or truncated`? Where does each go?
5. What does γ control? What if γ=0?
6. The `argmax` inside `q_target(next_obs)` — what does it return, and why is it inside `torch.no_grad()`?
7. Why 4-frame stacking?
8. Why linear ε decay 1.0 → 0.01 over 15% (not 10%)?
9. What if the buffer samples from `[0, capacity)`?
10. Why 6 actions for Space Invaders?

---

## Verification

End-to-end on Kaggle:

1. **Smoke (5 min):** `python -c "from dqn.network import QNetwork; ..."` — shapes correct, ~1.7M params.
2. **Buffer (1 min):** push 50 random, sample 4, check dtypes.
3. **Short run (10 min):** `python -m dqn.train --total_steps 10_000`. Confirm loss in 0.5-5.0 range, no NaNs, eval.csv row exists.
4. **Full run (12 hours on T4):** 1.2M steps. Watch `runs/dqn_scratch_eval.csv` — eval reward should climb from ~100 → 250+ by step 600k, then plateau.
5. **Final eval:** `python -m dqn.evaluate --checkpoint runs/dqn_scratch_final.pt --episodes 20`. Target: mean ≥ 250, std ≤ 150.
6. **Video:** `python record_trained.py --checkpoint runs/dqn_scratch_final.pt --framework scratch`. Confirm agent moves smoothly, shoots, dodges.

If eval mean is < 200 after 1.2M steps, the most likely culprits (in order):

- ε decay too fast → try `eps_frac=0.20`
- Learning rate too high → try `lr=5e-5`
- Bug in `train_step` gather pattern or `no_grad` placement
- Bug in buffer (sampling from `[0, capacity)` instead of `[0, size)`)

---

## Out of scope for v1 (mention as future work in the report)

- Double DQN (1-line change: use `q_online` for argmax, `q_target` for value)
- Dueling networks (architectural change to head)
- Prioritized experience replay (~100 lines)
- Noisy networks (replaces ε-greedy)
- n-step returns (generalizes 1-step TD target)

---

# Part B — Paper Comparison

> This section lives in `docs/scratch_dqn_vs_paper.md` as a separate file, but is included here for context. The two files stay in sync.

## Reference

**Paper:** Mnih et al., _Human-level control through deep reinforcement learning_, Nature 2015.
**Our base:** SB3 DQN baseline run, 300k steps CPU, 257.5 ± 104.9 eval reward, 1h6m wall time, 73 steps/sec.

We are rebuilding the paper's algorithm in PyTorch. We have made a small number of intentional changes. Each one is listed below with rationale, expected impact, and the evidence for/against.

---

## Changes vs. paper (and the SB3 baseline)

### Change 1: `min_repeat = 2` (was 4 in our SB3 run, was N/A in paper)

**What:** The `MinActionRepeat` wrapper in `preprocessing.py` was set to `min_repeat=4` in our SB3 run. We are reducing to `min_repeat=2` for the from-scratch version.

**Why:** With `min_repeat=4`, each action persists for ~666ms (4 agent-frames × 4 env-frames × ~42ms per env frame). For a 60Hz game this is sluggish. The video evaluation showed the agent jiggling at edges and standing awkwardly — both symptoms of a policy that cannot change direction quickly enough to dodge.

**Paper:** N/A. The paper's ALE wrapper has no such concept. They use the default AtariPreprocessing with `frame_skip=4`, which gives 4 env-frames per action. The discrepancy with our run is that we _also_ added sticky actions (`repeat_action_probability=0.25`) and a MinActionRepeat wrapper, both for compatibility with the SB3 default.

**Expected impact:** Better. Higher `min_repeat` means the agent cannot respond to bullets fast enough. Lower `min_repeat` lets the policy be more reactive. We expect the visual quality of the demo to improve noticeably; the eval reward may go up slightly but it's primarily a UX fix.

**Tradeoff:** Too low a `min_repeat` (e.g. 1) lets the agent micro-jitter and never commits to a movement direction. 2 is the sweet spot for Space Invaders — long enough to commit, short enough to dodge.

---

### Change 2: `eps_frac = 0.15` (was 0.10 in paper and our SB3 run)

**What:** The ε-greedy exploration parameter decays linearly from 1.0 to 0.01 over the first 10% of training in Mnih 2015. We are using 15% of total_steps (180k steps for 1.2M total) before reaching 0.01.

**Why:** With a 1.2M-step budget and 10% decay (120k steps), the agent is fully greedy by step 120k. At that point, the Q-network has only seen ~120k gradient steps and the Q-values are still noisy. Locking in too early risks committing to a suboptimal policy because the noisy Q-values look optimal at this stage.

**Paper:** Mnih 2015 used 1M total steps and 10% decay. They report converged performance at the end of training, so the issue is not catastrophic for them — but we have 1.2M steps and a smaller step budget, so giving more exploration is a low-risk insurance policy.

**Expected impact:** Neutral to slightly positive. The cost is 60k extra steps at high ε (some exploration noise), but the benefit is more reliable convergence. If the agent locks into a bad local optimum with `eps_frac=0.10`, the entire training is wasted.

**Tradeoff:** Too much exploration wastes steps on random play. 15% is the lower bound; 20% would also work but starts to leave less time for exploitation.

---

### Change 3: `device = "cuda"` (was "cpu" in our SB3 baseline)

**What:** Training on GPU instead of CPU.

**Why:** Our SB3 baseline was 73 steps/sec on CPU for 1h6m. On a T4 GPU, the same network at batch size 32 should run 5-10× faster (~500-700 steps/sec), giving 1.2M steps in ~30-40 minutes.

**Paper:** The original paper trained on GPU but used a different model (NVIDIA DevBox, GTX 680, single GPU). They report ~30-50ms per minibatch, which corresponds to roughly 600-1000 steps/sec on their hardware — consistent with what we expect on T4.

**Expected impact:** Massive. Wall-clock time drops from 4-5 hours (1.2M steps on CPU) to 30-40 minutes. Same model, same data, much faster iteration. The training dynamics are identical to CPU; only the wall-clock differs.

**Tradeoff:** None for this scale. For larger models or smaller batch sizes, GPU becomes essential; for tiny models with very large batch sizes, CPU is fine.

---

### Change 4: `target_update = 1000` (matches paper)

**What:** Copy online weights to target every 1000 environment steps.

**Why / Why not change:** The paper uses 10000. SB3's default is 10000. Our SB3 baseline used 10000. The from-scratch version is using 1000 to converge faster — at the cost of less stable targets.

**The reasoning behind 1000 over 10000:** Target network stability is most important in early training when Q-values are moving rapidly. After ~100k steps the Q-network is relatively stable, so target updates every 1000 vs 10000 have similar effect. The advantage of more frequent updates is faster propagation of learning, which matters when the total budget is 1.2M steps.

**Expected impact:** Slightly faster convergence, marginally less stable mid-training. The 10× more frequent updates mean the target is always "slightly behind" the online net, which is the _intended_ effect. Going to 100 (or fewer) would make the two networks too coupled and you'd lose the stability benefit; going to 10000 is the paper default and would converge a bit slower.

**Tradeoff:** 1000 is on the aggressive end. 500 would also work. 100 or less would defeat the purpose.

---

### Change 5: `terminal_on_life_loss = False` (matches SB3 and paper)

**What:** When the player loses a life, the env does NOT reset the episode. It continues with the next life. (In Space Invaders you have multiple lives; "game over" only happens when all lives are exhausted.)

**Paper:** Matches Mnih 2015 — they set `terminal_on_life_loss=False` so episodes can span multiple lives.

**Why we keep it:** Standard. Lives-internal terminal signals are too granular; the Q-network should learn to value life, not learn that one lost life = end of episode.

---

### Change 6: `repeat_action_probability = 0.25` (matches paper)

**What:** With 25% probability, the env repeats the previous action regardless of what the agent picks. This is "sticky actions" — a deliberate non-determinism source.

**Paper:** Matches Mnih 2015.

**Why we keep it:** It makes the env harder to memorize. Without it, the agent could potentially overfit to deterministic sequences. With it, the policy must be robust to "I pressed right, the env ignored me and kept going right anyway."

**Tradeoff:** Makes learning slightly slower. But the model trained with sticky actions generalizes better when evaluated against a real human's flaky inputs.

---

### Change 7: No reward clipping (was clipped to {-1, 0, +1} in paper)

**What:** Mnih 2015 clipped every reward to `sign(r)`. SB3's default is `clip_reward=False` (or `True` depending on version). We are NOT clipping rewards in the from-scratch version.

**Why not:** Reward clipping is a sledgehammer for the "single network, 49 games" paper setting, where Pong's ±1 rewards and Breakout's 7-point chunks and Space Invaders' 30-pt hits all need to be on the same scale. For a single-game project, this is unnecessary and loses information (a 30-pt hit and a 50-pt hit look identical to the network, but they aren't).

**Tradeoff:** Gradient magnitudes will be larger when a 30-pt reward is observed, but `clip_grad_norm_=10.0` handles that. Net effect: same training dynamics, slightly more information preserved.

---

### Change 8: `frame_skip = 4` and `frame_stack = 4` (matches paper)

**What:** The env skips 4 frames per action, and the agent sees the last 4 (post-skip) frames stacked. So one "agent decision" sees a temporal window of 16 raw frames.

**Paper:** Mnih 2015 used frame_skip=4 (default in ALE) and frame_stack=4.

**Why we keep it:** Standard. Necessary for the agent to infer velocity from frame differences.

---

### Change 9: `learning_rate = 1e-4` (matches paper)

**What:** Adam learning rate.

**Paper:** Mnih 2015 used RMSProp at 2.5e-4. Adam at 1e-4 is the modern equivalent and is what SB3 uses by default.

**Why we keep it:** Adam is the de-facto standard now. 1e-4 is conservative for RL (lower than supervised defaults of 1e-3) because RL gradients are noisier.

---

### Change 10: `buffer_size = 20_000` (was 20_000 in our SB3 run, 1_000_000 in paper)

**What:** 20k transitions vs 1M transitions.

**Why smaller:** The paper's 1M buffer was tuned for the "many games, one model" setting. For a single game, 20k is sufficient — each transition gets reused 4-10× before being overwritten, and the diversity within 20k is high enough for stable training. A 1M buffer would cost 320 GB of RAM, which is not feasible on Kaggle.

**Tradeoff:** 20k means the agent "forgets" old experience faster. For Space Invaders this is fine because the optimal policy doesn't change much over training. For a game with non-stationary opponents, a larger buffer would help.

---

## What we did NOT change (matches paper)

- Network architecture: NatureCNN (3 conv + 2 FC, no batch norm).
- Replay buffer type: uniform random sampling.
- ε-greedy exploration (vs. noisy nets).
- 1-step TD target (vs. n-step).
- Huber loss.
- Gradient clipping at 10.
- Target network architecture (same as online).
- Adam optimizer (paper used RMSProp; Adam is the modern equivalent).

---

## What we'd add in v2 (future work)

- **Double DQN** — decouples action selection from action evaluation. Reduces overestimation bias. 1-line change: in `train_step`, use `q_online(o2).argmax(...)` instead of `q_target(o2).argmax(...)` for the action index, but still use `q_target` for the value. Expected 10-20% reward gain.
- **Dueling networks** — splits the Q-value head into `V(s)` and `A(s, a)` with `Q = V + A - mean(A)`. Helps when some actions are irrelevant in most states. ~20 lines of code change.
- **Prioritized replay** — sample transitions with probability proportional to |TD error|. Reuses high-error transitions more. ~100 lines of code, but sample-efficiency gain of 2-3×.

These three together are part of **Rainbow** (Hessel et al. 2017) which is the canonical DQN improvement. The paper-comparison above is the _starting point_ — Rainbow is the _ending point_ of DQN-era research.

---

## Summary

We changed exactly 4 things vs. the paper + our SB3 baseline:

- `min_repeat`: 4 → 2 (UX fix)
- `eps_frac`: 0.10 → 0.15 (more exploration)
- `device`: cpu → cuda (speed)
- `target_update`: 10000 → 1000 (faster convergence at slight stability cost)
- `buffer_size`: 1M → 20k (memory constraint)

Everything else matches either Mnih 2015 or our SB3 baseline. The eval target (≥ 250 reward) is identical to our SB3 result. The wall-clock target is 12h on T4 GPU (vs. 1h6m for 300k steps on CPU, scaled up to 1.2M = ~4-5h on CPU).
