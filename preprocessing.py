from __future__ import annotations
import gymnasium as gym
# The "ALE/SpaceInvaders-v5" namespace is registered by ale_py, NOT by
# shimmy (shimmy only bridges gym<->gymnasium for the GymV21/V26 envs).
# ale_py's top-level __init__.py auto-registers on import, but on
# Python 3.12 + numpy 2.x it can race against gymnasium's first import
# and silently fail (you see a UserWarning at startup). Force the
# registration here so the ALE namespace is guaranteed to be populated
# before env_fixed() calls gym.make(...).
import ale_py  # noqa: F401  -- side-effect: calls register_v5_envs()
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
# FrameStack has moved in two ways across gymnasium releases:
#   (a) Module home:
#       0.26 - 0.29: gymnasium.wrappers.frame_stack.FrameStack
#       1.0+       : gymnasium.wrappers.FrameStackObservation (the module
#                    file was renamed to stateful_observation.py; there is
#                    NO frame_stack_observation sub-module).
#   (b) Constructor kwarg:
#       0.26 - 0.29: num_stack
#       1.0+       : num_frames
# Both (a) and (b) are handled below.
try:
    from gymnasium.wrappers import FrameStackObservation as FrameStack  # type: ignore[attr-defined]  # 1.0+
except ImportError:
    try:
        from gymnasium.wrappers import FrameStack  # type: ignore[attr-defined]  # 0.26 - 0.29
    except ImportError:
        from gymnasium.wrappers.frame_stack import FrameStack  # type: ignore[assignment]  # noqa: F401

# Detect which kwarg this gymnasium version expects, then expose a small
# adapter so the call sites below don't have to care.
import inspect as _inspect
_FRAME_STACK_KWARG = (
    "num_frames"
    if "num_frames" in _inspect.signature(FrameStack.__init__).parameters
    else "num_stack"
)


def _frame_stack(env, n: int):
    """Wrap env with FrameStack, using whichever kwarg name this gymnasium
    version expects. Centralized so we only inspect once at import time."""
    return FrameStack(env, **{_FRAME_STACK_KWARG: n})


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
  env = _frame_stack(env, 4)

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
    env = _frame_stack(env, 4)
    if seed is not None:
       env.action_space.seed(seed)
    return env
