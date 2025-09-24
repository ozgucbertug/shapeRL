"""Evaluation metrics for sand shaping."""

from __future__ import annotations

from typing import Callable, Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from tf_agents.environments import tf_py_environment

from shape_rl.terrain import HeightMap

EMD_MAX_POINTS = 512


def heightmap_pointcloud(heightmap: HeightMap) -> np.ndarray:
    surface = np.array(heightmap.map, dtype=np.float32, copy=False)
    h, w = surface.shape
    centered = surface - float(surface.mean())
    yy, xx = np.indices((h, w), dtype=np.float32)
    if h > 1:
        yy /= float(h - 1)
    if w > 1:
        xx /= float(w - 1)
    cloud = np.stack((xx, yy, centered), axis=-1)
    return cloud.reshape(-1, 3)


def chamfer_distance(pc_a: np.ndarray, pc_b: np.ndarray) -> float:
    if pc_a.size == 0 or pc_b.size == 0:
        return float('nan')
    tree_a = cKDTree(pc_a)
    tree_b = cKDTree(pc_b)
    dist_ab, _ = tree_a.query(pc_b, k=1)
    dist_ba, _ = tree_b.query(pc_a, k=1)
    return float(0.5 * (np.mean(dist_ab ** 2) + np.mean(dist_ba ** 2)))


def earth_movers_distance(pc_a: np.ndarray, pc_b: np.ndarray, max_points: int = EMD_MAX_POINTS) -> float:
    if pc_a.size == 0 or pc_b.size == 0:
        return float('nan')
    n_a = pc_a.shape[0]
    n_b = pc_b.shape[0]
    m = min(n_a, n_b, max_points)
    if m < 1:
        return float('nan')
    idx_a = np.linspace(0, n_a - 1, num=m, dtype=int)
    idx_b = np.linspace(0, n_b - 1, num=m, dtype=int)
    sample_a = pc_a[idx_a]
    sample_b = pc_b[idx_b]
    cost = cdist(sample_a, sample_b, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


def compute_eval(env_factory: Callable[[int | None], Any], policy, num_episodes: int = 10, base_seed: int | None = None) -> Dict[str, Any]:
    rmse_deltas: list[float] = []
    rmse_aucs: list[float] = []
    rmse_slopes: list[float] = []
    rel_improves: list[float] = []
    chamfers: list[float] = []
    chamfers_init: list[float] = []
    chamfer_deltas: list[float] = []
    chamfer_aucs: list[float] = []
    chamfer_slopes: list[float] = []
    emds: list[float] = []
    emds_init: list[float] = []
    emd_deltas: list[float] = []
    emd_aucs: list[float] = []
    emd_slopes: list[float] = []
    init_rmses: list[float] = []
    final_rmses: list[float] = []
    min_rmses: list[float] = []
    episode_lengths: list[float] = []
    returns: list[float] = []
    success_flags: list[float] = []

    for ep in range(num_episodes):
        env = env_factory(ep if base_seed is None else base_seed + ep)
        tf_env = tf_py_environment.TFPyEnvironment(env)
        time_step = tf_env.reset()
        diff0 = env._env_map.difference(env._target_map)
        rmse0 = np.sqrt(np.mean(diff0 ** 2))
        init_rmses.append(rmse0)
        env_cloud_init = heightmap_pointcloud(env._env_map)
        tgt_cloud_init = heightmap_pointcloud(env._target_map)
        chamfer_init = chamfer_distance(env_cloud_init, tgt_cloud_init)
        emd_init = earth_movers_distance(env_cloud_init, tgt_cloud_init)
        chamfers_init.append(chamfer_init)
        emds_init.append(emd_init)
        rmse_series = [rmse0]
        chamfer_series = [chamfer_init]
        emd_series = [emd_init]
        reward_series = []
        while not time_step.is_last():
            action_step = policy.action(time_step)
            batched_action = action_step.action
            time_step = tf_env.step(batched_action)
            diff = env._env_map.difference(env._target_map)
            rmse_series.append(np.sqrt(np.mean(diff ** 2)))
            env_cloud_step = heightmap_pointcloud(env._env_map)
            chamfer_series.append(chamfer_distance(env_cloud_step, tgt_cloud_init))
            emd_series.append(earth_movers_distance(env_cloud_step, tgt_cloud_init))
            try:
                reward_series.append(float(time_step.reward.numpy()))
            except Exception:
                pass

        rmse_initial = rmse_series[0]
        rmse_final = rmse_series[-1]
        rmse_min = min(rmse_series)
        rmse_delta = rmse_initial - rmse_final
        rel = rmse_delta / rmse_initial if rmse_initial > 0 else 0.0
        rmse_auc = np.trapz(rmse_series)
        steps = np.arange(len(rmse_series))
        rmse_slope = np.polyfit(steps, rmse_series, 1)[0]
        episode_len = max(len(rmse_series) - 1, 0)
        episode_return = float(np.sum(reward_series)) if reward_series else 0.0
        threshold_abs = getattr(env, '_error_threshold_abs', env._error_threshold)
        success = 1.0 if rmse_final <= threshold_abs else 0.0

        chamfer_final = chamfer_series[-1]
        chamfer_delta = chamfer_series[0] - chamfer_final
        chamfer_auc = np.trapz(chamfer_series)
        chamfer_slope = np.polyfit(steps, chamfer_series, 1)[0]

        emd_final = emd_series[-1]
        emd_delta = emd_series[0] - emd_final
        emd_auc = np.trapz(emd_series)
        emd_slope = np.polyfit(steps, emd_series, 1)[0]

        rmse_deltas.append(rmse_delta)
        rel_improves.append(rel)
        rmse_aucs.append(rmse_auc)
        rmse_slopes.append(rmse_slope)
        final_rmses.append(rmse_final)
        min_rmses.append(rmse_min)
        episode_lengths.append(episode_len)
        returns.append(episode_return)
        success_flags.append(success)
        chamfers.append(chamfer_final)
        chamfer_deltas.append(chamfer_delta)
        chamfer_aucs.append(chamfer_auc)
        chamfer_slopes.append(chamfer_slope)
        emds.append(emd_final)
        emd_deltas.append(emd_delta)
        emd_aucs.append(emd_auc)
        emd_slopes.append(emd_slope)

    metrics = {
        'init_rmse_mean': float(np.mean(init_rmses)) if init_rmses else 0.0,
        'rmse_delta_mean': float(np.mean(rmse_deltas)),
        'rel_improve_mean': float(np.mean(rel_improves)),
        'rmse_auc_mean': float(np.mean(rmse_aucs)),
        'rmse_slope_mean': float(np.mean(rmse_slopes)),
        'final_rmse_mean': float(np.mean(final_rmses)) if final_rmses else 0.0,
        'min_rmse_mean': float(np.mean(min_rmses)) if min_rmses else 0.0,
        'episode_length_mean': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        'return_mean': float(np.mean(returns)) if returns else 0.0,
        'success_rate': float(np.mean(success_flags)) if success_flags else 0.0,
        'chamfer_init_mean': float(np.mean(chamfers_init)) if chamfers_init else 0.0,
        'chamfer_final_mean': float(np.mean(chamfers)) if chamfers else 0.0,
        'chamfer_delta_mean': float(np.mean(chamfer_deltas)) if chamfer_deltas else 0.0,
        'chamfer_auc_mean': float(np.mean(chamfer_aucs)) if chamfer_aucs else 0.0,
        'chamfer_slope_mean': float(np.mean(chamfer_slopes)) if chamfer_slopes else 0.0,
        'emd_init_mean': float(np.mean(emds_init)) if emds_init else 0.0,
        'emd_final_mean': float(np.mean(emds)) if emds else 0.0,
        'emd_delta_mean': float(np.mean(emd_deltas)) if emd_deltas else 0.0,
        'emd_auc_mean': float(np.mean(emd_aucs)) if emd_aucs else 0.0,
        'emd_slope_mean': float(np.mean(emd_slopes)) if emd_slopes else 0.0,
        'init_rmse_list': init_rmses,
        'rmse_delta_list': rmse_deltas,
        'rel_improve_list': rel_improves,
        'rmse_auc_list': rmse_aucs,
        'rmse_slope_list': rmse_slopes,
        'final_rmse_list': final_rmses,
        'min_rmse_list': min_rmses,
        'episode_length_list': episode_lengths,
        'return_list': returns,
        'success_list': success_flags,
        'chamfer_init_list': chamfers_init,
        'chamfer_list': chamfers,
        'chamfer_delta_list': chamfer_deltas,
        'chamfer_auc_list': chamfer_aucs,
        'chamfer_slope_list': chamfer_slopes,
        'emd_init_list': emds_init,
        'emd_list': emds,
        'emd_delta_list': emd_deltas,
        'emd_auc_list': emd_aucs,
        'emd_slope_list': emd_slopes,
    }
    return metrics


__all__ = [
    'compute_eval',
    'chamfer_distance',
    'earth_movers_distance',
    'heightmap_pointcloud',
    'EMD_MAX_POINTS',
]
