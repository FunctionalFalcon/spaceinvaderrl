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

### Change 2: `eps_frac = 0.15` (was 0.10 in paper and our SB3 run)

**What:** The ε-greedy exploration parameter decays linearly from 1.0 to 0.01 over the first 10% of training in Mnih 2015. We are using 15% of `total_steps` (180k steps for 1.2M total) before reaching 0.01.

**Why:** With a 1.2M-step budget and 10% decay (120k steps), the agent is fully greedy by step 120k. At that point, the Q-network has only seen ~120k gradient steps and the Q-values are still noisy. Locking in too early risks committing to a suboptimal policy because the noisy Q-values look optimal at this stage.

**Paper:** Mnih 2015 used 1M total steps and 10% decay. They report converged performance at the end of training, so the issue is not catastrophic for them — but we have 1.2M steps and a smaller step budget, so giving more exploration is a low-risk insurance policy.

**Expected impact:** Neutral to slightly positive. The cost is 60k extra steps at high ε (some exploration noise), but the benefit is more reliable convergence. If the agent locks into a bad local optimum with `eps_frac=0.10`, the entire training is wasted.

**Tradeoff:** Too much exploration wastes steps on random play. 15% is the lower bound; 20% would also work but starts to leave less time for exploitation.

---

### Change 3: `device = "cuda"` (was "cpu" in our SB3 baseline)

**What:** Training on GPU instead of CPU.

**Why:** Our SB3 baseline was 73 steps/sec on CPU. On a T4 GPU, the same network at batch size 32 should run 5-10× faster (~500-700 steps/sec), giving 1.2M steps in 30-40 minutes.

**Paper:** The original paper trained on GPU (NVIDIA DevBox, GTX 680, single GPU). They report ~30-50ms per minibatch, which corresponds to roughly 600-1000 steps/sec on their hardware — consistent with what we expect on T4.

**Expected impact:** Massive. Wall-clock time drops from 4-5 hours (1.2M steps on CPU) to 30-40 minutes. Same model, same data, much faster iteration. The training dynamics are identical to CPU; only the wall-clock differs.

**Tradeoff:** None for this scale. For larger models or smaller batch sizes, GPU becomes essential; for tiny models with very large batch sizes, CPU is fine.

---

### Change 4: `target_update = 1000` (was 10000 in paper and our SB3 run)

**What:** Copy online weights to target every 1000 environment steps.

**Why:** The paper uses 10000. SB3's default is 10000. Our SB3 baseline used 10000. The from-scratch version uses 1000 to converge faster — at the cost of less stable targets.

The reasoning: target network stability is most important in early training when Q-values are moving rapidly. After ~100k steps the Q-network is relatively stable, so target updates every 1000 vs 10000 have similar effect. The advantage of more frequent updates is faster propagation of learning, which matters when the total budget is 1.2M steps.

**Expected impact:** Slightly faster convergence, marginally less stable mid-training. The 10× more frequent updates mean the target is always "slightly behind" the online net, which is the *intended* effect. Going to 100 (or fewer) would make the two networks too coupled and you'd lose the stability benefit; going to 10000 is the paper default and would converge a bit slower.

**Tradeoff:** 1000 is on the aggressive end. 500 would also work. 100 or less would defeat the purpose.

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

### Change 10: `buffer_size = 20_000` (was 20_000 in our SB3 run, 1_000_000 in paper)

**What:** 20k transitions vs 1M transitions.

**Why smaller:** The paper's 1M buffer was tuned for the "many games, one model" setting. For a single game, 20k is sufficient — each transition gets reused 4-10× before being overwritten, and the diversity within 20k is high enough for stable training. A 1M buffer would cost 320 GB of RAM, which is not feasible on Kaggle.

**Tradeoff:** 20k means the agent "forgets" old experience faster. For Space Invaders this is fine because the optimal policy doesn't change much over training. For a game with non-stationary opponents, a larger buffer would help.

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
- No reward clipping (matches our SB3 baseline).

**Improvements over Mnih 2015:**

We upgraded from vanilla DQN toward Rainbow (Hessel et al., 2017):

1. **Double DQN (DDQN)** — online net picks action, target net provides value. Reduces overestimation bias.
2. **Dueling networks** — V(s) + A(s,a) head instead of direct Q(s,a). Better action-value estimates.
3. **Noisy Nets** — learned weight noise replaces ε-greedy. State-dependent exploration that never decays.
4. **Prioritized Replay** — smart sampling by |TD error| via SumTree. 2-5× sample efficiency improvement.

Remaining Rainbow techniques (not yet implemented):
- **N-step Returns** — bootstrap over N steps instead of 1.
- **Categorical DQN (C51)** — learn the full return distribution.

---

## Implementation details

### Dueling + Noisy QNetwork
- File: [network.py](scratch/network.py)
- ~6.5M parameters (vs 1.7M for standard DQN)
- All FC layers use `NoisyLinear` (Fortunato et al., 2017)
- Dueling aggregation: `Q = V + A - mean(A)` (Wang et al., 2016)
- `reset_noise()` called after each training step

### Prioritized Replay
- File: [replay_buffer.py](scratch/replay_buffer.py)
- `SumTree`: O(log N) prioritized sampling and priority updates
- Priority: `P(i) ∝ |TD_error|^alpha` (alpha=0.6)
- IS weights: `w_i ∝ (N * P(i))^(-beta)`, beta ramps from 0.4 → 1.0
- New transitions get `max_priority` so they're replayed early

### Hyperparameters for v2 run
```
total_steps: 8,000,000
buffer_size: 100,000 (larger buffer for prioritized replay)
prio_alpha: 0.6
prio_beta: 0.4 → 1.0 (IS correction ramps over first 50% of training)
```

---

## Summary

**Architecture changes from vanilla DQN:**
- `QNetwork` → Dueling + Noisy (3 improvements combined)
- `ReplayBuffer` → `PrioritizedReplayBuffer` (SumTree-backed)
- `agent.train_step()` → adds IS weights to loss, updates priorities

**Hyperparameter changes:**
- `total_steps`: 2M → 8M
- `buffer_size`: 20k → 100k
- `save_freq`: 50k → 100k
- `eval_episodes`: 5 → 10
- `min_repeat`: 4 → 3 (preserved)
- `eps_frac`: 0.10 → 0.15 (preserved; Noisy Nets handle exploration)
- `device`: cpu → cuda (preserved)

**Previous results (vanilla DQN, 8M steps):**
- Best eval: 697 ± 232 at step 8M
- 2.7× the SB3 baseline

**Expected with Rainbow-level improvements:**
- 2-5× sample efficiency from Prioritized Replay
- Better exploration throughout training from Noisy Nets
- More accurate Q-values from Dueling
- Target: 1000+ eval reward on Space Invaders
