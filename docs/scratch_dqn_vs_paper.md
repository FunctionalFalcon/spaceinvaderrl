# From-Scratch DQN: Comparison vs. Mnih 2015 (and our SB3 baseline)

> Internal writeup. Not for the teacher yet — refine before submission.

## Reference

**Paper:** Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015.
**Our baseline:** SB3 DQN run, 300k steps CPU, 257.5 ± 104.9 eval reward, 1h6m wall time, 73 steps/sec.

We are rebuilding the paper's algorithm in PyTorch from scratch. We have made a small number of intentional changes. Each one is listed below with rationale, expected impact, and the evidence for/against.

The plan file (`dqn/PLAN.md`) has the build order. This file has the *why*.

---

## Changes vs. paper (and the SB3 baseline)

### Change 1: `min_repeat = 3` (was 4 in our SB3 run, N/A in paper)

**What:** The `MinActionRepeat` wrapper in `preprocessing.py` was set to `min_repeat=4` in our SB3 run. We use `min_repeat=3` for the from-scratch version.

**Why:** With `min_repeat=4`, each action persists for ~666ms (4 agent-frames × 4 env-frames × ~42ms per env frame). For a 60Hz game this is sluggish. The video evaluation showed the agent jiggling at edges and standing awkwardly — both symptoms of a policy that cannot change direction quickly enough to dodge bullets.

**Paper:** N/A. The paper's ALE wrapper has no such concept. They use default AtariPreprocessing with `frame_skip=4`, which gives 4 env-frames per action. The discrepancy with our run is that we *also* added sticky actions (`repeat_action_probability=0.25`) and a `MinActionRepeat` wrapper, both for compatibility with SB3 defaults.

**Expected impact:** Better. Higher `min_repeat` means the agent cannot respond to bullets fast enough. Lower `min_repeat` lets the policy be more reactive. 3 is a compromise between 2 (too jittery, agent can crash off edges) and 4 (too sluggish). The eval reward may improve; the main benefit is better visual quality.

**Tradeoff:** `min_repeat=2` risks edge crashes (agent zigzags to screen edge before committing to NOOP). `min_repeat=4` is too sluggish for dodging fast beams. 3 is the sweet spot — enough commitment to prevent edge crashes, fast enough to dodge.

---

### Change 2: `eps_frac = 0.15` — with ε-greedy exploration (not Noisy Nets)

**What:** The ε-greedy exploration parameter decays linearly from 1.0 to 0.01 over the first 15% of training. With 8M total steps, epsilon reaches 0.01 by step ~1.2M and stays there for the remainder of training.

**Why ε-greedy over Noisy Nets:** In Rainbow (Hessel et al., 2017), Noisy Nets works *because* it's combined with C51 (Distributional RL). C51 maintains a probability distribution over returns — the noise in the value estimate gets absorbed by the distribution's spread. Scalar DQN has no such cushion: the noise baked into NoisyLinear weights propagates directly into TD errors, which are then amplified through PER (prioritized sampling). This creates competing policy modes that cause the eval score to oscillate wildly (~100–300) while Q-values climb indefinitely.

ε-greedy cleanly separates exploration from the value function. The network learns accurate Q-values; the ε schedule handles exploration. No entanglement.

**Historical note:** We initially deployed Noisy Nets expecting the exploration benefits from Rainbow. After 4.7M steps of eval oscillation (Q-values: 3→160, eval: 99–289), the root cause was identified as the Noisy Nets / PER interaction.

**Expected impact:** Stable training without the Q-value / eval divergence. The agent converges smoothly to a policy rather than oscillating.

---

### Change 3: `device = "cuda"` (was "cpu" in our SB3 baseline)

**What:** Training on GPU instead of CPU.

**Why:** Our SB3 baseline was 73 steps/sec on CPU. On a T4 GPU, the same network at batch size 32 should run 5-10× faster (~500-700 steps/sec), giving 1.2M steps in 30-40 minutes.

**Paper:** The original paper trained on GPU (NVIDIA DevBox, GTX 680, single GPU). They report ~30-50ms per minibatch, which corresponds to roughly 600-1000 steps/sec on their hardware — consistent with what we expect on T4.

**Expected impact:** Massive. Wall-clock time drops from 4-5 hours (1.2M steps on CPU) to 30-40 minutes. Same model, same data, much faster iteration. The training dynamics are identical to CPU; only the wall-clock differs.

**Tradeoff:** None for this scale. For larger models or smaller batch sizes, GPU becomes essential; for tiny models with very large batch sizes, CPU is fine.

---

### Change 4: `target_update_tau = 0.005` (soft updates + periodic hard reset)

**What:** Instead of hard-copying online weights to target every N steps, we apply a soft update every training step: `θ_target = τ·θ_online + (1-τ)·θ_target` with τ=0.005. Additionally, a full hard-copy reset fires every 1M steps to break any value drift cycle.

**Why soft updates over hard copy:** Hard copies every 10K steps create sudden jumps in the target value, which can destabilize training. Soft updates with τ=0.005 make the target track the online net smoothly — the target is always "slightly behind" without sharp discontinuities. This is the DDPG/LunarLander style update, proven stable over long training runs.

**Why periodic hard reset:** Even with soft updates, Q-values can drift upward over millions of steps (the online net always chases its own predictions). A hard reset every 1M steps fully breaks the drift cycle and anchors the target to the current policy.

**Expected impact:** More stable training, no eval oscillation, no Q-value runaway. Target: smooth eval curve without the ~100-300 oscillation seen with hard-copy + Noisy Nets.

---

### Change 5: `terminal_on_life_loss = False` (matches SB3 and paper)

**What:** When the player loses a life, the env does NOT reset the episode. It continues with the next life. (In Space Invaders you have multiple lives; "game over" only happens when all lives are exhausted.)

**Paper:** Matches Mnih 2015.

**Why we keep it:** Standard. Lives-internal terminal signals are too granular; the Q-network should learn to value life, not learn that one lost life = end of episode.

---

### Change 6: `repeat_action_probability = 0.25` (matches paper)

**What:** With 25% probability, the env repeats the previous action regardless of what the agent picks. This is "sticky actions" — a deliberate non-determinism source.

**Paper:** Matches Mnih 2015.

**Why we keep it:** It makes the env harder to memorize. Without it, the agent could potentially overfit to deterministic sequences. With it, the policy must be robust to "I pressed right, the env ignored me and kept going right anyway."

**Tradeoff:** Makes learning slightly slower. But the model trained with sticky actions generalizes better when evaluated against a real human's flaky inputs.

---

### Change 7: No reward clipping (was clipped to {-1, 0, +1} in paper)

**What:** Mnih 2015 clipped every reward to `sign(r)`. SB3's default depends on version. We are NOT clipping rewards in the from-scratch version.

**Why not:** Reward clipping is a sledgehammer for the "single network, 49 games" paper setting, where Pong's ±1 rewards and Breakout's 7-point chunks and Space Invaders' 30-pt hits all need to be on the same scale. For a single-game project, this is unnecessary and loses information (a 30-pt hit and a 50-pt hit look identical to the network, but they aren't).

**Tradeoff:** Gradient magnitudes will be larger when a 30-pt reward is observed, but `clip_grad_norm_=10.0` handles that. Net effect: same training dynamics, slightly more information preserved.

---

### Change 8: `frame_skip = 4` and `frame_stack = 4` (matches paper)

**What:** The env skips 4 frames per action, and the agent sees the last 4 (post-skip) frames stacked. So one "agent decision" sees a temporal window of 16 raw frames.

**Paper:** Mnih 2015 used frame_skip=4 (default in ALE) and frame_stack=4.

**Why we keep it:** Standard. Necessary for the agent to infer velocity from frame differences.

---

### Change 9: `learning_rate = 1e-4` (matches paper's modern equivalent)

**What:** Adam learning rate.

**Paper:** Mnih 2015 used RMSProp at 2.5e-4. Adam at 1e-4 is the modern equivalent and is what SB3 uses by default.

**Why we keep it:** Adam is the de-facto standard now. 1e-4 is conservative for RL (lower than supervised defaults of 1e-3) because RL gradients are noisier.

---

### Change 10: `buffer_size = 100_000` (was 20_000 in our SB3 run, 1_000_000 in paper)

**What:** 100k transitions vs 20k (SB3) or 1M (paper). With prioritized replay, a larger buffer is beneficial: the priority-based sampling can draw from a wider diversity of experience.

**Why larger with PER:** Unlike uniform replay where old stale transitions hurt, PER reweights by TD error — old transitions that still have high error (e.g., a high-value action never revisited) stay in the replay distribution. 100k gives enough headroom for the SumTree to maintain a useful priority distribution across training.

**Tradeoff:** 100k means ~3 MB per array (float32), so ~15 MB total buffer. Well within GPU memory. The PER overhead (SumTree O(log N) operations) scales with buffer size, but 100k is negligible compared to the conv forward passes.

---

## What we did NOT change (matches paper)

- 1-step TD target (vs. n-step — future work).
- Huber loss.
- Gradient clipping at 10.
- Target network architecture (same as online).
- Adam optimizer (paper used RMSProp; Adam is the modern equivalent).
- `terminal_on_life_loss = False` (matches paper).
- `repeat_action_probability = 0.25` (matches paper).
- `frame_skip = 4` and `frame_stack = 4` (matches paper).
- No reward clipping.

**Improvements over Mnih 2015:**

We upgraded from vanilla DQN toward Rainbow (Hessel et al., 2017):

1. **Double DQN (DDQN)** — online net picks action, target net provides value. Reduces overestimation bias.
2. **Dueling networks** — V(s) + A(s,a) head instead of direct Q(s,a). Better action-value estimates.
3. **ε-greedy exploration** — standard Mnih 2015 approach with linear decay. Noisy Nets removed after causing oscillation with PER (see above).
4. **Prioritized Replay** — smart sampling by |TD error| via SumTree. 2-5× sample efficiency improvement.
5. **Soft target updates** — tau=0.005 per step + hard reset every 1M. Replaces hard-copy every 10K.

Remaining Rainbow techniques (not yet implemented):
- **N-step Returns** — bootstrap over N steps instead of 1.
- **Categorical DQN (C51)** — learn the full return distribution.

---

## Implementation details

### DuelingDQN (ε-greedy)
- File: [network.py](scratch/network.py) — `DuelingDQN` class
- Same NatureCNN backbone as paper, with two heads: V(s) and A(s,a)
- Standard `nn.Linear` layers — exploration is handled by ε-greedy, not weight noise
- Dueling aggregation: `Q = V + A - mean(A)` (Wang et al., 2016)
- Also available: `QNetwork` class (Dueling + NoisyLinear) behind `--use-noisy` flag for comparison

### Prioritized Replay
- File: [replay_buffer.py](scratch/replay_buffer.py)
- `SumTree`: O(log N) prioritized sampling and priority updates
- Priority: `P(i) ∝ |TD_error|^alpha` (alpha=0.4 — tuned down from 0.6 for more stability)
- IS weights: `w_i ∝ (N * P(i))^(-beta)`, beta ramps from 0.4 → 1.0 over first 75% of training
- New transitions get `max_priority` so they're replayed early

### Hyperparameters for current run
```
total_steps: 8,000,000
buffer_size: 100,000
lr: 1e-4 (Adam)
prio_alpha: 0.4  (less aggressive than 0.6)
prio_beta: 0.4 → 1.0 (over first 75% of training)
target_update_tau: 0.005 (soft update every step)
target_hard_reset_freq: 1,000,000 (hard copy every 1M to break drift)
eps_frac: 0.15 (epsilon decays over first 15% = 1.2M steps)
```

---

## Summary

**Architecture changes from vanilla DQN:**
- `QNetworkLegacy` → `DuelingDQN` (Dueling architecture, standard Linear layers)
- `ReplayBuffer` → `PrioritizedReplayBuffer` (SumTree-backed)
- `agent.train_step()` → adds IS weights to loss, updates priorities
- Target update: hard copy every 10K → soft update every step (tau=0.005) + hard reset every 1M

**Hyperparameter changes from v1:**
- `lr`: 2.5e-4 → 1e-4 (more stable for long runs)
- `prio_alpha`: 0.6 → 0.4 (less aggressive prioritization)
- `prio_beta_frac`: 0.5 → 0.75 (slower IS correction ramp)
- `target_update_tau`: 0.005 (soft updates — no hard jump discontinuities)
- `target_hard_reset_freq`: 1,000,000 (breaks value drift cycle)
- `total_steps`: 8M (preserved)
- `buffer_size`: 100k (preserved)
- `min_repeat`: 3 (preserved)
- `eps_frac`: 0.15 (preserved; ε-greedy, not Noisy Nets)

**Previous results (Noisy Nets + PER, 8M steps):**
- Best eval: 361 (step 7M, run cancelled)
- Problem: eval oscillated 99–289 while Q-values climbed 3→160+
- Root cause: Noisy Nets + PER without C51 — exploration noise amplified through prioritized sampling

**Expected with ε-greedy:**
- Stable training without Q-value / eval divergence
- Smooth convergence to a fixed policy
- Target: ≥ 304.2 mean eval reward (beat SB3 DQN baseline)
