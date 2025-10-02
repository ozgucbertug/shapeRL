"""Evaluation metrics for sand shaping."""

from __future__ import annotations

from typing import Callable, Dict, Any
from contextlib import suppress

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from tf_agents.environments import tf_py_environment

from shape_rl.terrain import HeightMap


EMD_MAX_POINTS = 512
WASSERSTEIN_MAX_POINTS = 1024


def normalized_auc(series: list[float] | np.ndarray) -> float:
    s = np.asarray(series, dtype=np.float64)
    if s.size <= 1:
        return 0.0
    denom = (s.size - 1) * max(abs(float(s[0])), 1e-12)
    return float(np.trapz(s) / denom)


def _block_sum(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return arr.astype(np.float64, copy=True)
    h, w = arr.shape
    h2 = (h // stride) * stride
    w2 = (w // stride) * stride
    if h2 == 0 or w2 == 0:
        return arr.astype(np.float64, copy=True)
    trimmed = arr[:h2, :w2]
    reshaped = trimmed.reshape(h2 // stride, stride, w2 // stride, stride)
    return reshaped.sum(axis=(1, 3)).astype(np.float64)


def _grad_lap_rms(diff: np.ndarray) -> tuple[float, float]:
    gy, gx = np.gradient(diff)
    grad_rms = float(np.sqrt(np.mean(gy * gy + gx * gx)))
    # 5-point Laplacian
    lap = (np.roll(diff, 1, axis=0) + np.roll(diff, -1, axis=0) +
           np.roll(diff, 1, axis=1) + np.roll(diff, -1, axis=1) - 4.0 * diff)
    lap_rms = float(np.sqrt(np.mean(lap * lap)))
    return grad_rms, lap_rms


def _prepare_wasserstein_masses(diff: np.ndarray, stride: int, mass_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Positive mass = surplus sand, Negative mass = deficit (as positive values)
    pos = np.maximum(diff, 0.0)
    neg = np.maximum(-diff, 0.0)
    if stride > 1:
        pos = _block_sum(pos, stride)
        neg = _block_sum(neg, stride)
    h, w = pos.shape
    if h == 0 or w == 0:
        return (np.zeros((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=np.float64))
    # Coordinates are block centers in normalized units
    yy, xx = np.indices((h, w), dtype=np.float64)
    # Normalize by original grid size approximation (after pooling)
    x_norm = (xx + 0.5) / max(w, 1)
    y_norm = (yy + 0.5) / max(h, 1)

    # Flatten and filter
    a_mass = pos.ravel()
    b_mass = neg.ravel()
    a_coords = np.stack((x_norm.ravel(), y_norm.ravel()), axis=-1)
    b_coords = a_coords  # same grid

    a_mask = a_mass > mass_threshold
    b_mask = b_mass > mass_threshold
    return a_coords[a_mask], a_mass[a_mask], b_coords[b_mask], b_mass[b_mask]


def _sinkhorn_wasserstein(a: np.ndarray, b: np.ndarray, xa: np.ndarray, xb: np.ndarray,
                          reg: float = 0.05, max_iter: int = 200, tol: float = 1e-8) -> float:
    # Ensure proper distributions
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    sa = float(a.sum())
    sb = float(b.sum())
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    # Normalize to probability masses, remember total mass for scaling later
    a /= sa
    b /= sb

    # Cost matrix (Euclidean distance in normalized pixel units)
    C = cdist(xa, xb, metric='euclidean')
    # Stabilize by clipping extreme costs to avoid underflow in exp
    C = np.asarray(C, dtype=np.float64)
    reg = float(max(reg, 1e-6))
    K = np.exp(-C / reg)

    u = np.ones_like(a) / max(a.size, 1)
    v = np.ones_like(b) / max(b.size, 1)
    for _ in range(max_iter):
        Kv = K @ v + 1e-12
        u_new = a / Kv
        KTu = K.T @ u_new + 1e-12
        v_new = b / KTu
        if np.max(np.abs(u_new - u)) < tol and np.max(np.abs(v_new - v)) < tol:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new

    P = (u[:, None] * K) * v[None, :]
    # Scale back by geometric mass (take the mean of sa and sb which should be equal for mean-centered diffs)
    total_mass = 0.5 * (sa + sb)
    return float(np.sum(P * C) * total_mass)


def wasserstein_distance_2d(env_or_map: HeightMap | np.ndarray,
                             target_map: HeightMap | np.ndarray | None = None,
                             stride: int = 4,
                             reg: float = 0.05,
                             max_points: int = WASSERSTEIN_MAX_POINTS,
                             mass_threshold: float = 1e-8) -> float:
    """Compute a 2D (entropic-regularized) Wasserstein-1 distance between surplus and deficit.

    If `target_map` is provided, we compute `diff = env - target` using mean-centering
    semantics of `HeightMap.difference`. If only a single array is provided, it's
    treated directly as `diff`.
    """
    if target_map is not None:
        if isinstance(env_or_map, HeightMap) and isinstance(target_map, HeightMap):
            diff = env_or_map.difference(target_map)
        else:
            env_arr = env_or_map.map if isinstance(env_or_map, HeightMap) else np.asarray(env_or_map, dtype=np.float64)
            tgt_arr = target_map.map if isinstance(target_map, HeightMap) else np.asarray(target_map, dtype=np.float64)
            # Mean-center both like HeightMap.difference does
            diff = (env_arr - float(np.mean(env_arr))) - (tgt_arr - float(np.mean(tgt_arr)))
    else:
        diff = np.asarray(env_or_map, dtype=np.float64)

    xa, a, xb, b = _prepare_wasserstein_masses(diff, stride, mass_threshold)
    if a.size == 0 or b.size == 0:
        return 0.0

    # Optionally cap to top-k masses to keep the problem small
    if a.size > max_points:
        idx = np.argpartition(-a, max_points - 1)[:max_points]
        xa, a = xa[idx], a[idx]
    if b.size > max_points:
        idx = np.argpartition(-b, max_points - 1)[:max_points]
        xb, b = xb[idx], b[idx]

    return _sinkhorn_wasserstein(a, b, xa, xb, reg=reg)


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


def compute_eval(env_factory: Callable[[int | None], Any], policy, num_episodes: int = 10,
                 base_seed: int | None = None,
                 w2_stride: int = 4, w2_reg: float = 0.05,
                 w2_max_points: int = WASSERSTEIN_MAX_POINTS) -> Dict[str, Any]:
    rmse_deltas: list[float] = []
    rmse_aucs_norm: list[float] = []
    rmse_slopes: list[float] = []
    mae_deltas: list[float] = []
    mae_aucs_norm: list[float] = []
    mae_slopes: list[float] = []
    w2_deltas: list[float] = []
    w2_aucs_norm: list[float] = []
    w2_slopes: list[float] = []

    init_rmses: list[float] = []
    final_rmses: list[float] = []
    init_maes: list[float] = []
    final_maes: list[float] = []
    init_w2s: list[float] = []
    final_w2s: list[float] = []

    grad_final_list: list[float] = []
    lap_final_list: list[float] = []
    grad_auc_norm_list: list[float] = []
    lap_auc_norm_list: list[float] = []

    episode_lengths: list[float] = []
    returns: list[float] = []
    success_flags: list[float] = []

    for ep in range(num_episodes):
        env = env_factory(ep if base_seed is None else base_seed + ep)
        tf_env = tf_py_environment.TFPyEnvironment(env)
        try:
            time_step = tf_env.reset()
            diff0 = env._env_map.difference(env._target_map)
            rmse0 = float(np.sqrt(np.mean(diff0 ** 2)))
            mae0 = float(np.mean(np.abs(diff0)))
            w20 = wasserstein_distance_2d(env._env_map, env._target_map,
                                          stride=w2_stride, reg=w2_reg,
                                          max_points=w2_max_points)
            grad0, lap0 = _grad_lap_rms(diff0)

            init_rmses.append(rmse0)
            init_maes.append(mae0)
            init_w2s.append(w20)

            rmse_series = [rmse0]
            mae_series = [mae0]
            w2_series = [w20]
            grad_series = [grad0]
            lap_series = [lap0]
            reward_series: list[float] = []

            while not time_step.is_last():
                action_step = policy.action(time_step)
                batched_action = action_step.action
                time_step = tf_env.step(batched_action)

                diff = env._env_map.difference(env._target_map)
                rmse_series.append(float(np.sqrt(np.mean(diff ** 2))))
                mae_series.append(float(np.mean(np.abs(diff))))
                w2_series.append(wasserstein_distance_2d(env._env_map, env._target_map,
                                                         stride=w2_stride, reg=w2_reg,
                                                         max_points=w2_max_points))
                g, l = _grad_lap_rms(diff)
                grad_series.append(g)
                lap_series.append(l)
                try:
                    reward_series.append(float(time_step.reward.numpy()))
                except Exception:
                    pass
        finally:
            with suppress(Exception):
                tf_env.close()
            with suppress(Exception):
                env.close()

        rmse_initial = rmse_series[0]
        rmse_final = rmse_series[-1]
        mae_initial = mae_series[0]
        mae_final = mae_series[-1]
        w2_initial = w2_series[0]
        w2_final = w2_series[-1]

        steps = np.arange(len(rmse_series))
        rmse_slope = float(np.polyfit(steps, rmse_series, 1)[0])
        mae_slope = float(np.polyfit(steps, mae_series, 1)[0])
        w2_slope = float(np.polyfit(steps, w2_series, 1)[0])

        rmse_auc_n = normalized_auc(rmse_series)
        mae_auc_n = normalized_auc(mae_series)
        w2_auc_n = normalized_auc(w2_series)
        grad_auc_n = normalized_auc(grad_series)
        lap_auc_n = normalized_auc(lap_series)

        episode_len = max(len(rmse_series) - 1, 0)
        episode_return = float(np.sum(reward_series)) if reward_series else 0.0
        threshold_abs = getattr(env, '_error_threshold_abs', env._error_threshold)
        success = 1.0 if rmse_final <= threshold_abs else 0.0

        rmse_deltas.append(rmse_initial - rmse_final)
        rmse_aucs_norm.append(rmse_auc_n)
        rmse_slopes.append(rmse_slope)
        final_rmses.append(rmse_final)

        mae_deltas.append(mae_initial - mae_final)
        mae_aucs_norm.append(mae_auc_n)
        mae_slopes.append(mae_slope)
        final_maes.append(mae_final)

        w2_deltas.append(w2_initial - w2_final)
        w2_aucs_norm.append(w2_auc_n)
        w2_slopes.append(w2_slope)
        final_w2s.append(w2_final)

        grad_final_list.append(float(grad_series[-1]))
        lap_final_list.append(float(lap_series[-1]))
        grad_auc_norm_list.append(grad_auc_n)
        lap_auc_norm_list.append(lap_auc_n)

        episode_lengths.append(episode_len)
        returns.append(episode_return)
        success_flags.append(success)

    metrics = {
        # RMSE
        'init_rmse_mean': float(np.mean(init_rmses)) if init_rmses else 0.0,
        'final_rmse_mean': float(np.mean(final_rmses)) if final_rmses else 0.0,
        'rmse_delta_mean': float(np.mean(rmse_deltas)),
        'rmse_auc_norm_mean': float(np.mean(rmse_aucs_norm)),
        'rmse_slope_mean': float(np.mean(rmse_slopes)),
        'init_rmse_list': init_rmses,
        'final_rmse_list': final_rmses,
        'rmse_delta_list': rmse_deltas,
        'rmse_auc_norm_list': rmse_aucs_norm,
        'rmse_slope_list': rmse_slopes,

        # MAE
        'init_mae_mean': float(np.mean(init_maes)) if init_maes else 0.0,
        'final_mae_mean': float(np.mean(final_maes)) if final_maes else 0.0,
        'mae_delta_mean': float(np.mean(mae_deltas)) if mae_deltas else 0.0,
        'mae_auc_norm_mean': float(np.mean(mae_aucs_norm)) if mae_aucs_norm else 0.0,
        'mae_slope_mean': float(np.mean(mae_slopes)) if mae_slopes else 0.0,
        'init_mae_list': init_maes,
        'final_mae_list': final_maes,
        'mae_delta_list': mae_deltas,
        'mae_auc_norm_list': mae_aucs_norm,
        'mae_slope_list': mae_slopes,

        # Wasserstein-1 (2D, entropic-regularized)
        'w2_init_mean': float(np.mean(init_w2s)) if init_w2s else 0.0,
        'w2_final_mean': float(np.mean(final_w2s)) if final_w2s else 0.0,
        'w2_delta_mean': float(np.mean(w2_deltas)) if w2_deltas else 0.0,
        'w2_auc_norm_mean': float(np.mean(w2_aucs_norm)) if w2_aucs_norm else 0.0,
        'w2_slope_mean': float(np.mean(w2_slopes)) if w2_slopes else 0.0,
        'w2_init_list': init_w2s,
        'w2_final_list': final_w2s,
        'w2_delta_list': w2_deltas,
        'w2_auc_norm_list': w2_aucs_norm,
        'w2_slope_list': w2_slopes,

        # Gradient/Laplacian diagnostics
        'grad_rms_final_mean': float(np.mean(grad_final_list)) if grad_final_list else 0.0,
        'lap_rms_final_mean': float(np.mean(lap_final_list)) if lap_final_list else 0.0,
        'grad_rms_auc_norm_mean': float(np.mean(grad_auc_norm_list)) if grad_auc_norm_list else 0.0,
        'lap_rms_auc_norm_mean': float(np.mean(lap_auc_norm_list)) if lap_auc_norm_list else 0.0,
        'grad_rms_final_list': grad_final_list,
        'lap_rms_final_list': lap_final_list,
        'grad_rms_auc_norm_list': grad_auc_norm_list,
        'lap_rms_auc_norm_list': lap_auc_norm_list,

        # Episode stats
        'episode_length_mean': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        'return_mean': float(np.mean(returns)) if returns else 0.0,
        'success_rate': float(np.mean(success_flags)) if success_flags else 0.0,
        'episode_length_list': episode_lengths,
        'return_list': returns,
        'success_list': success_flags,
    }
    return metrics


__all__ = [
    'compute_eval',
    'heightmap_pointcloud',
    'normalized_auc',
    'wasserstein_distance_2d',
    'EMD_MAX_POINTS',
    'WASSERSTEIN_MAX_POINTS',
    'chamfer_distance',
    'earth_movers_distance',
]
