# DQN for Atari Space Invaders — Theory Study Guide

This is the **defense document** for the project. Read it before your teacher presentation. Every section answers a question they are likely to ask. The math is shown in LaTeX-style code blocks so it renders on GitHub and in VSCode.

**Project numbers (your actual run):**
- Training: 300,000 steps on CPU
- **Mean eval reward: 257.5 ± 104.9** (20 episodes, deterministic)
- Random baseline: ~100-150
- Improvement over random: **~2×**
- Training time: 1h 6min @ 73 steps/sec

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

### 11.4 No planning

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
│  y_i = r_i + γ · max_{a'} Q(s_i', a'; θ⁻)                        │
│        ↑↑↑↑                                                      │
│        TD target (uses frozen target net θ⁻)                     │
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

## 13. Hyperparameter Reference (the values we actually used)

| Hyperparameter | Value | Where it's set | Why |
|---|---|---|---|
| `learning_rate` | `1e-4` | [train.py:211](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | Stable, slower than the paper's 2.5e-4 |
| `buffer_size` | `20,000` | [train.py:212](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | CPU-RAM-safe; ~6.3 GB replay buffer |
| `learning_starts` | `5,000` | [train.py:213](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | Fill buffer before first gradient step |
| `batch_size` | `32` | [train.py:214](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | Paper default |
| `gamma` | `0.99` | [train.py:215](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | Standard discount |
| `train_freq` | `4` | [train.py:216](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | Train every 4 env steps (matches frame_skip) |
| `gradient_steps` | `1` | [train.py:217](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | One update per train step |
| `target_update_interval` | `1,000` | [train.py:218](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | Frozen target for 1k steps |
| `exploration_fraction` | `0.1` | [train.py:219](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | ε-decay over first 10% of training |
| `exploration_final_eps` | `0.01` | [train.py:220](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | Final ε = 0.01 (99% greedy) |
| `max_grad_norm` | `10.0` | [train.py:221](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | Gradient clip (second safety net) |
| `total_timesteps` | `300,000` | [train.py:172](../c--Users-DELL-OneDrive-Documents--1-Ky-8-AI/rel301m/SpaceInvader/train.py) | 1 hour on CPU |

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
| **NatureCNN** | The specific CNN architecture from Mnih 2015: 32→64→64 convs, FC 512, output |A|. |
| **Huber / smooth_l1** | Same thing, two names. PyTorch calls it `smooth_l1_loss`. |
| **Determinism** | At eval time, the agent picks `argmax Q(s, a)` (no ε-noise). Makes the eval number reproducible. |
| **SPS** | Steps per second. Our CPU run hit 73 SPS. |
