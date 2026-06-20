from __future__ import annotations

import gymnasium as gym
import ale_py
import cv2

# Register ALE environments with gymnasium
gym.register_envs(ale_py)

# Create the raw environment
env: gym.Env = gym.make(
    "ALE/SpaceInvaders-v5",
    frameskip=1,
    repeat_action_probability=0.25,
)

# Print spaces
print("obs space:", env.observation_space)
print("action space:", env.action_space)

# get_action_meanings lives on the underlying AtariEnv; Pylance doesn't know about it
print("actions:", env.unwrapped.get_action_meanings())  # type: ignore[attr-defined]

# Reset and save the first frame
obs, info = env.reset(seed=0)
cv2.imwrite("sanity_frame.png", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
print(f"frame shape: {obs.shape}, dtype: {obs.dtype}, "
      f"min: {obs.min()}, max: {obs.max()}")

# Run 200 random actions and accumulate reward
total = 0.0
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total += float(reward)  # explicit cast silences Pylance
    if terminated or truncated:
        obs, info = env.reset()

print("random 200-step total reward:", total)
env.close()
