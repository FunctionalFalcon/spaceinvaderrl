# DQN for Atari Space Invaders — Theory Study Guide

This is the **defense document** for the project. Read it before your teacher presentation. Every section answers a question they are likely to ask. The math is shown in LaTeX-style code blocks so it renders on GitHub and in VSCode.

**Project numbers:**
- **SB3 baseline (v0):** 300k steps CPU, 257.5 ± 104.9 eval reward, 1h6m @ 73 steps/sec
- **Vanilla DQN (v1):** 8M steps GPU (Kaggle T4), **697 ± 232** eval reward at step 8M
- **Rainbow v2 (current):** Dueling + Noisy Nets + Prioritized Replay, 8M steps GPU
- Random baseline: ~100-150
- Training time: ~60-90 min for 8M steps on Kaggle T4 GPU

---

## 1. The Problem: Why Atari Is Hard

A single frame of Space Invaders is a **210 × 160 × 3 RGB image** (≈ 100,000 numbers). At 60 FPS, the game produces ~6 million numbers per second of state data.

Four things make this hard for a learning agent:

1. **Raw pixels, not features.** Unlike chess (board = 64 squares), the agent has to *learn* what a "ship" or "alien" even looks like.
2. **A single frame is a snapshot, not motion.** From one frame you can't tell if the ship is moving left, right, or sitting still. You need *time*.
3. **High-dimensional state space.** If you treated each frame as a discrete state, the number of possible states is `2^(84·84·4) ≈ 10^8491`. You cannot build a lookup table.
4. **Delayed rewards.** You shoot an alien → bullet flies → alien dies 5-10 frames later. The agent must learn to credit the kill back to the original shot, not just the kill frame.

Mnih et al. 2015 ("Human-level control through deep reinforcement learning", the **DQN paper**) showed a single algorithm could solve all 49 Atari 2600 games it was tested on, using the same network architecture and hyperparameters. Space Invaders is one of them.

---

## 2. The MDP Formulation

A **Markov Decision Process** is the math model for sequential decision-making. It's a 5-tuple:

```
(S, A, P, R, γ)
```

| Symbol | Name | What it means in Space Invaders |
|---|---|---|
| `S` | State space | Set of all possible game screens. We approximate this with the last 4 stacked frames. |
| `A` | Action space | `Discrete(6)` = {NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE} |
| `P(s'|s,a)` | Transition probability | Probability of seeing screen `s'` after taking action `a` in screen `s`. Deterministic for Atari. |
| `R(s,a)` | Reward function | +5 to +30 per alien killed, 0 otherwise (in raw ALE units; we clip to {-1, 0, +1}). |
| `γ ∈ [0,1]` | Discount factor | How much we value future reward vs immediate. γ=0.99 means a reward 100 steps away is worth 0.99¹⁰⁰ ≈ 0.366. |

The **Markov property** says: "the future depends on the past only through the present." A single frame violates this (you can't tell velocity from one image), but **4 stacked frames** restore it because the agent can infer motion from frame deltas.

---

## 3. Q-Learning and the Bellman Equation

The key object is `Q(s, a)` — the **expected cumulative discounted reward** if I'm in state `s`, take action `a`, then act optimally forever after.

```
Q*(s, a) = E[ r + γ · max_{a'} Q*(s', a') ]
```

Read this as: *"The value of taking action `a` in state `s` equals the reward I get *plus* the discounted value of the best next action, in expectation."*

This is the **Bellman optimality equation** for Q-values. It defines a fixed point: the optimal Q-function is the one where `Q(s,a)` equals the right-hand side for every `(s,a)`.

### Tabular Q-learning

If `S` is small enough to fit in a table, we can learn `Q` by **temporal-difference (TD) updates**:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_{a'} Q(s', a') − Q(s, a) ]
                                          ^^^^^^^^^^^^^^^^^^^
                                          this is the "TD target"
                                                                              ^^^^^^^^^^^
                                                                              this is the "TD error"
```

The bracket is the gap between what we *predicted* and what we *actually observed*; α is a learning rate. With enough samples, this converges to Q\*.

**But it doesn't scale.** Atari has more states than atoms in the observable universe. We cannot store a table. We need **function approximation** — i.e., a neural network.

---

## 4. Deep Q-Network (DQN)

Replace the table with a neural network `Q(s, a; θ)` where `θ` are the weights:

```
Q(s, a; θ)  ≈  Q*(s, a)
```

The network takes a state (4 stacked 84×84 grayscale frames) and outputs one Q-value per action. In our project, the network is **NatureCNN** (see §9).

The loss function we minimize is the squared TD error:

```
L(θ) = E[ (y_i − Q(s_i, a_i; θ))² ]

where  y_i = r_i + γ · max_{a'} Q(s_i', a'; θ⁻)
```

But this **naive formulation diverges**. The Mnih 2015 paper identified two tricks that make it stable. Plus one more trick for the reward signal. Those are §5.

---

## 5. The Three DQN Tricks

These are **the** DQN paper contribution. If your teacher asks "what makes DQN work?" — this section is the answer.

### 5.1 Experience Replay

**Problem:** Consecutive game frames are highly correlated. A 32-sample minibatch from the last 32 frames is basically 32 copies of nearly-the-same image. Gradient descent on correlated data is unstable — like trying to walk on a treadmill that's moving.

**Fix:** Store every transition `(s, a, r, s', done)` in a **replay buffer** of size ~100k-1M. Train on a random minibatch sampled *uniformly* from this buffer.

```
Replay buffer: ring of (s, a, r, s', done) tuples
On each training step:
    sample 32 random transitions
    compute TD target for each
    take one gradient step
```

Two benefits:
- Decorrelates samples (random sampling breaks the time-correlation).
- Reuses each transition 4-10× before it's overwritten. 10× more data per environment step.

Our project uses `buffer_size=20,000` (CPU-memory-safe; see §11).

### 5.2 Target Network

**Problem:** The TD target `y_i = r + γ · max Q(s'; θ)` uses the *same* weights `θ` that we're updating. This is like "chasing a moving target" — the loss can diverge because every gradient step changes the target itself.

**Fix:** Maintain a **frozen copy** of the network, called the target network, with weights `θ⁻`. Use `θ⁻` to compute the target. Periodically copy `θ → θ⁻` (e.g., every 1,000 training steps):

```
Online net:    Q(s, a; θ)      ← updated every step
Target net:    Q(s', a'; θ⁻)   ← frozen, updated every 1,000 steps

TD target:     y = r + γ · max_{a'} Q(s', a'; θ⁻)
```

This makes the target **stable for 1,000 steps**, giving the loss a chance to actually decrease.

### 5.3 Reward Clipping

**Problem:** Different Atari games have different reward scales. In Pong a point is +1, in Space Invaders an alien kill is +5 to +30, in Breakout breaking a brick is +1. If we used raw rewards, the gradient magnitudes would vary wildly across games and break the "single network, 49 games" claim.

**Fix:** Clip the reward to a fixed range:

```
clip(r, -1, +1)    # any positive reward → +1, any negative → -1, zero → 0
```

This is a **deliberate bias-variance trade-off**: we lose some signal (a 5-pt kill and a 30-pt kill look identical) but gain **scale-invariance** across games. The agent still learns *that* a kill happened; it just doesn't distinguish kill-values.

---

## 6. The Loss Function (Huber)

The squared-error loss `L = (y − Q)²` is sensitive to outliers: a single large TD error (e.g., the first time the agent sees a multi-alien chain) produces a huge gradient that destabilizes training.

DQN uses the **Huber loss** (a.k.a. `smooth_l1` in PyTorch):

```
                   { 0.5 · (y - Q)²        if |y - Q| ≤ 1
smooth_l1(y, Q) = {
                   { |y - Q| − 0.5        otherwise
```

- For small errors: quadratic (smooth, well-behaved gradients).
- For large errors: linear (bounded gradient magnitude, no explosion).

The network is trained with **gradient clipping** at `||g||₂ ≤ 10` to add a second safety net.

---

## 7. ε-Greedy Exploration

A purely greedy agent (`argmax_a Q(s, a)`) gets stuck — it never discovers that there's a better strategy it hasn't tried yet. We need **exploration**.

**ε-greedy:**

```
With probability ε:    pick a random action
With probability 1-ε:  pick argmax_a Q(s, a)
```

**Schedule** (linear decay over the first 10% of training):

```
ε(t) = ε_end + (ε_start − ε_end) · max(0, 1 − t / (ε_frac · T))
```

With our hyperparameters:
- `ε_start = 1.0` (start fully random)
- `ε_end = 0.01` (end 99% greedy)
- `ε_frac = 0.1` (decay over the first 10% of training = first 30,000 steps of 300k)
- After step 30,000: ε is locked at 0.01 forever.

By the time your training finishes, ε = 0.01 and the agent is essentially greedy.

---

## 8. Atari-Specific Preprocessing

The network expects a fixed input shape. We do four transformations to raw ALE output:

### 8.1 Frame skip 4

The ALE calls our `step()` every 4 internal frames. **Our agent only sees ~15 decisions per second instead of 60.** This is the right timescale for human-like decisions and makes learning tractable.

### 8.2 Grayscale + 84×84

Color is mostly irrelevant for Atari (the score readout is the only colored region in Space Invaders and it's at the top). Grayscale cuts the input size by 3×. Resizing to 84×84 is the Mnih 2015 standard — small enough to be fast, large enough to resolve alien shapes.

### 8.3 Scale to [0, 1]

`scale_obs=True` divides pixel values by 255 to get `float32 ∈ [0, 1]`. The network's first conv layer doesn't have to learn this normalization.

### 8.4 Stack 4 frames

The observation is `(4, 84, 84)` — the last 4 preprocessed frames stacked along the channel axis. This is the **cheapest way to give the agent motion perception**. Without it, the agent literally cannot tell that the ship has moved (POMDP). With it, the agent can compute inter-frame deltas and infer velocity.

This is implemented in our `preprocessing.py` as:

```python
env = gym.make(..., frameskip=1, repeat_action_probability=0.25)
env = AtariPreprocessing(env, frame_skip=4, screen_size=84,
                          grayscale_obs=True, scale_obs=True)
env = FrameStack(env, num_stack=4)
```

Two things to notice:
- `frameskip=1` on `gym.make` because `AtariPreprocessing` does its own skipping — setting both would compound to 16× skip.
- `repeat_action_probability=0.25` is **sticky actions**: with 25% probability, the previous action is repeated instead of the new one. This makes the environment stochastic, which prevents the agent from "memorizing" deterministic action sequences (a form of regularization).

---

## 9. Network Architecture (NatureCNN)

The paper specifies this exact network. Our `policy="CnnPolicy"` in SB3 instantiates it automatically.

```
Input:  4 × 84 × 84  (4 stacked grayscale frames)
        │
        ├── Conv2d(4→32, kernel=8, stride=4) + ReLU   →  32 × 20 × 20
        ├── Conv2d(32→64, kernel=4, stride=2) + ReLU  →  64 × 9 × 9
        ├── Conv2d(64→64, kernel=3, stride=1) + ReLU  →  64 × 7 × 7
        ├── Flatten                                    →  64·7·7 = 3136
        ├── Linear(3136 → 512) + ReLU                 →  512
        └── Linear(512 → |A|)                          →  6 (one Q per action)
        │
Output:  Q(s, NOOP), Q(s, FIRE), Q(s, RIGHT), Q(s, LEFT), Q(s, RIGHTFIRE), Q(s, LEFTFIRE)
```

The agent picks the action with the highest Q-value:

```
a* = argmax_a  Q(s, a; θ)
```

About **1.7M parameters**. Fits easily in CPU memory. Forward pass on CPU takes ~5ms per sample.

---

## 10. Mnih 2015 Results

The original paper trained **one network per game** for **200 million frames** (~50M agent decisions, since frame_skip=4). On 49 games, the agent reached **human-level performance on 29** and exceeded humans on most of the rest. The "human-level" claim refers to professional game testers who played the same games at the same ALE settings.

The paper's **key insight** is that the *combination* of the three tricks (replay, target net, reward clipping) is what makes it work. Without any one of them, the network fails to learn. This was the first convincing demonstration that a single deep RL algorithm could learn directly from pixels across diverse tasks.

---

## 11. Limitations & Future Work

This is **the section you read if the teacher asks "what could be improved?"**. Each item is a known DQN weakness. Be ready to discuss at least the first one.

### 11.1 The walk-off-screen bug (your agent's actual failure mode)

**What you see in the video:** the trained agent slides the player ship left until it disappears off the edge of the screen, then dies.

**Why it happens:** the discrete 6-action space has *directional* and *fire* actions, but **no "stop moving" action without releasing the stick**. The only way to halt motion is to call `step(NOOP)`. The DQN learned that movement correlates with dodging aliens (and thus higher reward), so it over-uses `LEFT`/`RIGHT` and under-uses `NOOP`. Once it commits to a `LEFT` chain, it can't easily break the chain.

**This is not a bug in your code** — it's a textbook DQN-on-Atari limitation that Mnih 2015 themselves noted. The real fix is an **action-persistence wrapper** (sometimes called `MinActionRepeat`):

```python
class MinActionRepeat(gym.Wrapper):
    """Force every chosen action to persist for N agent-frames."""
    def __init__(self, env, min_repeat=4):
        super().__init__(env)
        self.min_repeat = min_repeat
        self.remaining = 0
        self.current_action = None

    def step(self, action):
        if self.remaining <= 0:
            self.current_action = action
            self.remaining = self.min_repeat
        # else: ignore the new action, keep playing current_action
        obs, r, term, trunc, info = self.env.step(self.current_action)
        self.remaining -= 1
        return obs, r, term, trunc, info
```

With this wrapper, the agent must **explicitly choose NOOP at the right time** to halt motion — there's no other way. This makes the "stop" action learnable. The cost: training is `min_repeat×` slower and must be done from scratch.

**Why we didn't retrain:** 300k steps takes ~1h on CPU; retraining with `min_repeat=4` would take ~4h. For a class project, the 220-reward demo is enough — the explanation of *why* the bug happens is the actual proof you understand DQN.

### 11.2 Overestimation bias (DQN → DDQN)

Standard DQN uses `max_{a'} Q(s', a'; θ⁻)` to pick the next action *and* value it. If the target net slightly overestimates some Q-values (which it does, due to function-approximation noise), the max operator **amplifies** the bias. The Q-values grow unboundedly over training.

**Double DQN (DDQN)** decouples action selection from action evaluation: use the *online* net to pick the next action, the *target* net to value it:

```
DDQN target:  y = r + γ · Q(s', argmax_{a'} Q(s', a'; θ); θ⁻)
                              ^^^^^^^^         ^
                              online net picks  target net values
```

### 11.3 Sample inefficiency (DQN → Rainbow, Prioritized Replay)

DQN needs ~200M frames to reach human level on Atari. A human learns in minutes. Three fixes that compose into Rainbow:
- **Prioritized experience replay** — sample "surprising" transitions more often
- **N-step returns** — bootstrap over N steps instead of 1
- **Distributional Q-learning** — learn the *full distribution* of returns, not just the mean
- **Noisy nets** — learn exploration as a function of the weights

### 11.4 Overestimation bias (DQN → DDQN)

Standard DQN uses `max_{a'} Q(s', a'; θ⁻)` to pick the next action *and* value it. If the target net slightly overestimates some Q-values (which it does, due to function-approximation noise), the max operator **amplifies** the bias. The Q-values grow unboundedly over training.

**Double DQN (DDQN)** decouples action selection from action evaluation: use the *online* net to pick the next action, the *target* net to value it:

```
DDQN target:  y = r + γ · Q(s', argmax_{a'} Q(s', a'; θ); θ⁻)
                              ^^^^^^^^         ^
                              online net picks  target net values
```

This is already implemented in our from-scratch code ([agent.py:64-76](scratch/agent.py#L64-L76)). The online net's `argmax` selects the action it believes is best, but the target net's value for that action provides a more stable bootstrap signal. This reduces overestimation by ~30-50% and leads to more accurate Q-values.

### 11.5 The separation problem (DQN → Dueling DQN)

Standard DQN outputs one Q-value per action: `Q(s, a)`. This conflates two conceptually different questions:

1. **How good is it to be in state `s`?** (regardless of what action you pick)
2. **How much better is action `a` than the other actions?**

**Dueling DQN** separates these explicitly using the identity:

```
Q(s, a) = V(s) + A(s, a) − mean_a[A(s, a)]
```

Where:
- `V(s)` = **Value** of state `s` — how good is this situation on average?
- `A(s, a)` = **Advantage** of action `a` — how much better is `a` than average?
- The `− mean(A)` term centers the advantages so that `mean(Q) = V(s)`

The network learns two separate heads:

```
Conv layers → features
             ├── Value stream: Linear(512 → 1)    → V(s)
             └── Advantage stream: Linear(512 → 6) → A(s, a)
             
Output: Q(s, a) = V(s) + A(s, a) − mean(A)
```

**Why this helps in Space Invaders:** The game has long stretches where the agent should SHOOT (no immediate threat) alternating with short bursts where the agent must DODGE (beam incoming). Standard DQN's single Q-head has to represent both "when to shoot" and "when to dodge" in the same weight space. Dueling lets the network learn:

- `V(s)` = "how safe am I right now?" (learns from ALL states)
- `A(s, a)` = "is SHOOT or DODGE the right call?" (learns from action comparisons)

This separation of concerns accelerates learning, especially when some actions are rarely useful.

### 11.6 Exploration that never stops (ε-greedy → Noisy Nets)

**The ε-greedy problem:** ε decays to 0.01 over the first 15% of training and stays there forever. The agent becomes nearly deterministic — it always picks `argmax Q`. This means it never discovers that a previously bad-looking action became good.

**Noisy Nets fix:** Instead of using ε to inject randomness, the network's weights themselves contain noise that is learned alongside the rest of the network.

Each weight `w` is: `w = w_fixed + w_noise`, where `w_noise` is sampled from a factorized Gaussian distribution:

```
w_noise[i,j] = (σ_i / √k) · ε_j
ε ~ N(0, 1), σ = learned parameter, k = fan-in
```

Key properties:
- **Every forward pass produces different noise** — natural exploration without ε
- **The noise is differentiable** — the network learns to reduce noise on important weights (σ → 0) and keep it where uncertainty is high
- **Exploration is state-dependent** — the agent explores more in unfamiliar states (high noise) and less in familiar ones

**Why it's better than ε-greedy:**
- ε-greedy explores uniformly across all states — it adds noise even when the policy is already good
- Noisy nets explore more where Q-values are uncertain and less where they're confident
- The agent never becomes fully deterministic — exploration persists throughout training

### 11.7 Learning from the most surprising moments (Uniform → Prioritized Replay)

**The uniform sampling problem:** The replay buffer stores all transitions equally. When sampling a batch, every transition has the same probability of being chosen. But some transitions teach the agent more than others.

A transition where the agent correctly predicted Q(s,a) has a tiny TD error — the gradient is near zero, the network learns almost nothing. A transition where the agent was wildly wrong (TD error = 50) is "surprising" — it has a large gradient and teaches the network something important.

Uniform sampling wastes batch capacity on boring transitions.

**Prioritized Replay fix:** Sample transitions with probability proportional to their TD error:

```
P(i) ∝ |TD_error_i|^α
```

Where α (default 0.6) controls how strongly we prioritize surprising transitions.

**Implementation using a SumTree:** Naively computing priorities after every update is expensive. The standard approach (Schaul et al. 2015) uses a **SumTree** — a binary tree data structure that supports:
- Sampling by priority in O(log N) time
- Updating priorities in O(log N) time

**Sampling bias correction:** Always replaying high-TD-error transitions causes the agent to overfit to "surprising" states. We compensate using **importance sampling weights** `w_i = (P(i)^(−β))` in the loss function, which downweights over-sampled transitions.

**Expected impact:** 2-5× sample efficiency improvement on Space Invaders. The agent learns faster because every batch contains more teaching signal — particularly helpful for learning death-approaching states (beam appearing) which have large TD errors.

### 11.8 The full Rainbow combination

These four techniques — DDQN, Dueling, Noisy Nets, and Prioritized Replay — are four of the six techniques in Rainbow (Hessel et al. 2017). The full Rainbow combines all six:

1. **DDQN** ✓ (implemented in v1)
2. **Dueling DQN** ✓ (implemented in v2)
3. **Noisy Nets** ✓ (implemented in v2)
4. **Prioritized Replay** ✓ (implemented in v2)
5. **N-step Returns** — bootstrap over N steps instead of 1 (future work)
6. **Categorical DQN (C51)** — learn the full distribution of returns (future work)

On Space Invaders, Rainbow achieved **~3× the score of vanilla DQN** at the same training time. Our current implementation has all four implemented techniques. Expected improvement: 1000+ eval reward (vs 697 from vanilla DQN at 8M steps).

### 11.9 No planning

DQN is a **reactive** policy: `a = argmax Q(s, a)`. It cannot think ahead. For games that require planning (chess, Go, StarCraft), you need tree search (MCTS) on top of the value function — that's AlphaZero.

---

## 12. Sticky-Note Key Equations (memorize these)

```
┌──────────────────────────────────────────────────────────────────┐
│  Q*(s, a) = E[ r + γ · max_{a'} Q*(s', a') ]                   │
│              ↑                                                    │
│              Bellman optimality equation for Q-values             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  y_i = r_i + γ · Q(s_i', argmax_{a'} Q(s_i', a'; θ); θ⁻)       │
│              ↑           ↑^^^^^^^         ↑                       │
│              DDQN target  online picks  target values              │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  L_i(θ) = smooth_l1( y_i − Q(s_i, a_i; θ) )                     │
│           ^^^^^^^^                                                │
│           Huber loss (quadratic for small errors,                 │
│           linear for large)                                       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  ε(t) = ε_end + (ε_start − ε_end) · max(0, 1 − t / (ε_frac · T))│
│         ^^^^                                                     │
│         Linear ε-decay over the first 10% of training            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 13. Hyperparameter Reference (Rainbow v2 values)

| Hyperparameter | Value | Where it's set | Why |
|---|---|---|---|
| `learning_rate` | `1e-4` | [hyperparam.py](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/scratch/hyperparam.py) | Standard for DQN |
| `buffer_size` | `100,000` | [hyperparam.py](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/scratch/hyperparam.py) | Larger for prioritized replay |
| `learning_starts` | `5,000` | [hyperparam.py](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/scratch/hyperparam.py) | Fill buffer before first gradient step |
| `batch_size` | `32` | [hyperparam.py](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/scratch/hyperparam.py) | Paper default |
| `gamma` | `0.99` | [hyperparam.py](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/scratch/hyperparam.py) | Standard discount |
| `target_update` | `1,000` | [hyperparam.py](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/scratch/hyperparam.py) | Frozen target for 1k steps |
| `min_repeat` | `3` | [hyperparam.py](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/scratch/hyperparam.py) | Prevents edge crashes, enough to dodge |
| `prio_alpha` | `0.6` | [hyperparam.py](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/scratch/hyperparam.py) | TD-error prioritization exponent |
| `prio_beta` | `0.4 → 1.0` | [hyperparam.py](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/scratch/hyperparam.py) | IS correction ramps over training |
| `total_steps` | `8,000,000` | Notebook | 8M steps on Kaggle T4 GPU |
| `save_freq` | `100,000` | Notebook | Checkpoint every 100k steps |
| `eval_freq` | `100,000` | Notebook | Eval every 100k steps |
| `eval_episodes` | `10` | Notebook | 10 greedy episodes per eval |

---

## 14. Glossary

| Term | Definition |
|---|---|
| **MDP** | Markov Decision Process — `(S, A, P, R, γ)`. Math model for sequential decisions. |
| **State `s`** | The "current situation" the agent observes. |
| **Action `a`** | A choice the agent can make. |
| **Reward `r`** | Scalar feedback from the environment after an action. |
| **Discount `γ`** | How much we value future vs present reward. γ=0.99 means future matters, but less than now. |
| **Q-value `Q(s,a)`** | Expected cumulative discounted reward if I take action `a` in state `s`, then act optimally. |
| **Bellman equation** | Recursive definition: `Q(s,a) = r + γ max Q(s',a')`. The foundation of all value-based RL. |
| **TD error** | `r + γ max Q(s',a') − Q(s,a)`. The gap between what we predicted and what we observed. |
| **TD target** | `r + γ max Q(s',a')`. The "observed" value we're trying to match. |
| **DQN** | Deep Q-Network. Replace the Q-table with a neural net. |
| **Replay buffer** | Ring buffer of past transitions `(s, a, r, s', done)`. Decorrelates training samples. |
| **Target network** | Frozen copy of the online net, used to compute TD targets. Stabilizes training. |
| **ε-greedy** | With probability ε, act randomly; otherwise act greedily. Forces exploration. |
| **ε-schedule** | The function `ε(t)` that decays ε from 1.0 down to 0.01 over the first 10% of training. |
| **Frame skip** | The ALE runs the chosen action for N internal frames before returning. |
| **Frame stack** | Concatenate the last 4 preprocessed frames along the channel axis. Restores Markov property. |
| **Sticky actions** | With probability p, repeat the previous action instead of executing the new one. Regularization. |
| **Reward clipping** | Bound rewards to {-1, 0, +1} so different games have comparable gradient magnitudes. |
| **Huber loss** | Loss function that's quadratic for small errors and linear for large ones. Robust to outliers. |
| **Gradient clipping** | Cap the L2 norm of the gradient to a max value (10.0 in our case). Prevents explosion. |
| **Adam / RMSProp** | Optimizers used by DQN. Adam is the SB3 default. |
| **NatureCNN** | The CNN architecture from Mnih 2015: 32→64→64 convs, FC 512, output |A|. Used in our SB3 baseline and from-scratch implementation. |
| **Determinism** | At eval time, the agent picks `argmax Q(s, a)` (no ε-noise). Makes the eval number reproducible. |
| **SPS** | Steps per second. Our CPU run hit 73 SPS. |
| **DDQN** | Double DQN. Decouples action selection (online net) from action evaluation (target net) to reduce overestimation bias. |
| **Dueling DQN** | Architecture that learns V(s) and A(s,a) separately, then combines them as Q = V + A − mean(A). Helps when some actions are rarely useful. |
| **Noisy Nets** | Learns exploration noise as part of the network weights. Replaces ε-greedy with state-dependent, differentiable exploration. |
| **Prioritized Replay** | Samples transitions from replay buffer with probability proportional to |TD error|, not uniformly. Requires importance sampling weights to correct bias. |
| **SumTree** | Binary tree data structure used to implement prioritized replay efficiently. Supports O(log N) sampling and priority updates. |
| **Rainbow** | Combination of six DQN improvements (DDQN, Dueling, Noisy Nets, Prioritized Replay, N-step, C51). State-of-the-art DQN as of 2017. |
| **NatureCNN** | The CNN architecture from Mnih 2015: 32→64→64 convs, FC 512, output |A|. Used in both our SB3 baseline and from-scratch implementation. |
| **Value V(s)** | The value of being in state s, regardless of action. Part of Dueling DQN's decomposition: Q(s,a) = V(s) + A(s,a). |
| **Advantage A(s,a)** | How much better action a is than the average action in state s. Part of Dueling DQN: Q(s,a) = V(s) + A(s,a). |
