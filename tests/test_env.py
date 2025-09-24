import numpy as np
import pytest

pytest.importorskip("tf_agents")

from shape_rl.envs import SandShapingEnv


def test_env_reset_and_step_shapes():
    env = SandShapingEnv(width=32, height=32, patch_width=32, patch_height=32,
                         max_steps=3, seed=123, debug=True)

    ts0 = env.reset()
    assert ts0.observation.shape == (32, 32, 5)

    action = np.zeros(env.action_spec().shape, dtype=np.float32)
    ts1 = env.step(action)
    assert ts1.observation.shape == (32, 32, 5)
    assert isinstance(ts1.reward, float)
    assert hasattr(env, '_last_reward_terms')
    assert 'reward_clipped' in env._last_reward_terms
