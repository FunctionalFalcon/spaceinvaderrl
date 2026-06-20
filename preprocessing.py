from __future__ import annotations
import gymnasium as gym
# shimmy's plugin entry point races against gymnasium's first import on
# Python 3.12 + numpy 2.x. Force-register the ALE envs at module load.
# We import the registration submodule DIRECTLY (not `import shimmy`) to
# avoid triggering shimmy/__init__.py's top-level `import gym` side
# effect, which loads the unmaintained gym 0.26.2 and its numpy-1.x
# compiled cv2 wrapper — that crashes with AttributeError: _ARRAY_API
# not found on numpy 2.x.
import importlib
_shimmy_reg = importlib.import_module("shimmy.registration")
_shimmy_reg.register_gymnasium_envs()
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack


def make_env(env_id: str = "ALE/SpaceInvaders-v5", seed: int | None = None, render_mode: str | None = None) -> gym.Env:
  """Create a wrapped Atari env with DQN-friendly preprocessing.

  Pipeline:
      gym.make(...)  ->  AtariPreprocessing  ->  FrameStackObservation

  Args:
      env_id: ALE environment id (default Space Invaders v5).
      seed: random seed for action sampling (used during eval).
      render_mode: None | "rgb_array" | "human".
                    "rgb_array" is needed if you want to record videos later.
  """
    
  # raw env - set frameskip = 1 because AP does its own skippping
  env = gym.make(
      env_id,
      frameskip=1,
      repeat_action_probability=0.25,
      render_mode = render_mode
  )

  # 1 - AtariPreprocessing:
  # grayscale, 84x84, skip 4 frames, max-pool flickers

  env = AtariPreprocessing(
      env,
      frame_skip=4,
      screen_size=84,
      grayscale_obs=True,
      scale_obs=True, # rescale pixel values to [0, 1]
      terminal_on_life_loss=False, # keep long eps
  )

  # 2 - FrameStackObservation:
  # stack last 4 frames to give the agent a sense of motion
  env = FrameStack(env, num_stack=4)

  if seed is not None:
    env.action_space.seed(seed)
  

  return env



"""
  Once the agent picked an action, its stuck in that state for
  like, atleast 'min_repeat'
  After that, the agent can pick a new action
  
  Forces the agent to deliberately pick noop for 'min repeat'
  Prevent the runaway where the agent commits to movement
  action and cant release 
  """

class MinActionRepeat(gym.ActionWrapper):
    def __init__(self, env, min_repeat: int = 4):
        super().__init__(env)
        self.min_repeat = min_repeat
        self._counter = 0
        self._held_action = None

    def action(self,action):
       # First call, or counter expired: take the new action and reset counter
        if self._held_action is None or self._counter >= self.min_repeat:
          self._held_action = action
          self._counter = 0
        
        self._counter += 1
        return self._held_action


def env_fixed(env_id: str = "ALE/SpaceInvaders-v5",
              seed: int | None = None,
              render_mode: str | None = None,
              min_repeat: int = 4) -> gym.Env:
    """
    Wrapper order:
    gym.make -> MinActionRepeat -> AtariPreprocessing -> FrameStack

    min_repeat: num of agent-decisions each action
    must persist.

    min_repeat = 4 -> ~666ms of held action.
    Short enough to feel responsive, long enough to prevent runaway chains
    """
    env = gym.make(
      env_id,
      frameskip = 1, # AP have skipping
      repeat_action_probability = 0.25,
      render_mode = render_mode,
    )
    # MAR raps the env so it operates on agent-decisions
    # not the post-4x frame-skip macro-actions.
    env = MinActionRepeat(env, min_repeat = min_repeat)
    env = AtariPreprocessing(
       env,
       frame_skip = 4,
       screen_size = 84,
       grayscale_obs = True,
       scale_obs = True,
       terminal_on_life_loss = False,
    )
    env = FrameStack(env, num_stack = 4)
    if seed is not None:
       env.action_space.seed(seed)
    return env