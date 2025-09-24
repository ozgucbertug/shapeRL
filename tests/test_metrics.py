import math

import numpy as np
import pytest

pytest.importorskip("tf_agents")

from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy

from shape_rl.metrics import (
    heightmap_pointcloud,
    chamfer_distance,
    earth_movers_distance,
    compute_eval,
)
from shape_rl.envs import SandShapingEnv


def test_pointcloud_distances_zero_for_identical_maps():
    grid = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    class DummyHeightMap:
        def __init__(self, data):
            self.map = data

    hm_a = DummyHeightMap(grid)
    hm_b = DummyHeightMap(grid.copy())

    pc_a = heightmap_pointcloud(hm_a)
    pc_b = heightmap_pointcloud(hm_b)

    chamfer = chamfer_distance(pc_a, pc_b)
    emd = earth_movers_distance(pc_a, pc_b)

    assert math.isclose(chamfer, 0.0, abs_tol=1e-6)
    assert math.isclose(emd, 0.0, abs_tol=1e-6)


def test_compute_eval_returns_expected_keys():
    def factory(seed):
        return SandShapingEnv(width=32, height=32, patch_width=32, patch_height=32,
                              max_steps=2, seed=seed, debug=False)

    sample_env = tf_py_environment.TFPyEnvironment(factory(None))
    policy = random_tf_policy.RandomTFPolicy(sample_env.time_step_spec(), sample_env.action_spec())

    metrics = compute_eval(factory, policy, num_episodes=1, base_seed=0)

    required_keys = {
        'init_rmse_mean', 'final_rmse_mean', 'rmse_delta_mean',
        'rmse_auc_mean', 'rmse_slope_mean',
        'chamfer_init_mean', 'chamfer_final_mean', 'chamfer_delta_mean',
        'chamfer_auc_mean', 'chamfer_slope_mean',
        'emd_init_mean', 'emd_final_mean', 'emd_delta_mean',
        'emd_auc_mean', 'emd_slope_mean',
        'rmse_delta_list', 'chamfer_delta_list', 'emd_delta_list'
    }
    missing = required_keys.difference(metrics.keys())
    assert not missing, f"Missing metrics: {missing}"
