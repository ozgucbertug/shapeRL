"""Evaluation metrics for sand shaping."""

from __future__ import annotations

import os
import tensorflow as tf
from typing import Callable, Dict, Any, Sequence
from contextlib import suppress

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_policy as _py_policy

from tqdm.auto import tqdm

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

    rmse_series_list: list[list[float]] = []
    mae_series_list: list[list[float]] = []
    w2_series_list: list[list[float]] = []

    for ep in range(num_episodes):
        env = env_factory(ep if base_seed is None else base_seed + ep)
        is_py_policy = isinstance(policy, _py_policy.PyPolicy)
        tf_env = None
        try:
            if is_py_policy:
                time_step = env.reset()
            else:
                tf_env = tf_py_environment.TFPyEnvironment(env)
                time_step = tf_env.reset()
            diff0 = env._env_map.difference(env._target_map)
            rmse0 = float(np.sqrt(np.mean(diff0 ** 2)))
            mae0 = float(np.mean(np.abs(diff0)))
            w20 = wasserstein_distance_2d(env._env_map, env._target_map,
                                          stride=w2_stride, reg=w2_reg,
                                          max_points=w2_max_points)

            rmse_series = [rmse0]
            mae_series = [mae0]
            w2_series = [w20]

            while not time_step.is_last():
                action_step = policy.action(time_step)
                action = action_step.action
                if is_py_policy:
                    time_step = env.step(action)
                else:
                    time_step = tf_env.step(action)

                diff = env._env_map.difference(env._target_map)
                rmse_series.append(float(np.sqrt(np.mean(diff ** 2))))
                mae_series.append(float(np.mean(np.abs(diff))))
                w2_series.append(wasserstein_distance_2d(env._env_map, env._target_map,
                                                         stride=w2_stride, reg=w2_reg,
                                                         max_points=w2_max_points))
        finally:
            with suppress(Exception):
                (tf_env.close() if tf_env is not None else None)
            with suppress(Exception):
                env.close()

        rmse_series_list.append(rmse_series)
        mae_series_list.append(mae_series)
        w2_series_list.append(w2_series)

    def _stack_pad_nan(series_list):
        if not series_list:
            return None
        m = max(len(s) for s in series_list)
        if m <= 0:
            return None
        arr = np.full((len(series_list), m), np.nan, dtype=np.float64)
        for i, s in enumerate(series_list):
            si = np.asarray(s, dtype=np.float64)
            arr[i, :si.size] = si
        return arr
    rmse_arr = _stack_pad_nan(rmse_series_list)
    mae_arr = _stack_pad_nan(mae_series_list)
    w2_arr = _stack_pad_nan(w2_series_list)
    rmse_mean = np.nanmean(rmse_arr, axis=0).tolist() if rmse_arr is not None else []
    mae_mean = np.nanmean(mae_arr, axis=0).tolist() if mae_arr is not None else []
    w2_mean = np.nanmean(w2_arr, axis=0).tolist() if w2_arr is not None else []

    return {
        'rmse_series_mean': rmse_mean,
        'mae_series_mean': mae_mean,
        'w2_series_mean': w2_mean,
    }


def summarize_metric_series(series: Sequence[float], pos_tol: float = 1e-9) -> Dict[str, float]:
    """
    Derive scalar summary statistics from a per-step metric series.
    Returns deltas, slopes, normalized AUC, relative improvement, and the
    fraction of steps with positive (decreasing) improvements.
    """
    fallback = {
        'initial': -1.0,
        'final': -1.0,
        'delta': -1.0,
        'slope': -1.0,
        'auc_norm': -1.0,
        'pos_improve_frac': -1.0,
        'relative_improvement': -1.0,
    }

    arr = np.asarray(series, dtype=np.float64)
    if arr.size == 0:
        return fallback
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return fallback

    start_idx = int(np.argmax(finite_mask))
    trimmed = arr[start_idx:]
    contiguous_values: list[float] = []
    for value in trimmed:
        if np.isfinite(value):
            contiguous_values.append(float(value))
        else:
            break
    values = np.asarray(contiguous_values, dtype=np.float64)
    if values.size == 0:
        return fallback

    steps = start_idx + np.arange(values.size, dtype=np.float64)
    initial = float(values[0])
    final = float(values[-1])
    delta = initial - final
    slope = float(np.polyfit(steps, values, 1)[0]) if values.size >= 2 else 0.0
    auc_norm = normalized_auc(values.tolist())
    diffs = np.diff(values)
    pos_improve_frac = float(np.mean(diffs < -pos_tol)) if diffs.size > 0 else 0.0
    rel_improvement = delta / max(abs(initial), 1e-6)
    return {
        'initial': initial,
        'final': final,
        'delta': delta,
        'slope': slope,
        'auc_norm': auc_norm,
        'pos_improve_frac': pos_improve_frac,
        'relative_improvement': rel_improvement,
    }


# --- Helper: log per-step RMSE/MAE/W2 curves as native TensorBoard scalars in a separate run ---
def log_eval_metric_curves(metrics: dict, logdir: str, run_name: str, run_prefix: str = 'eval_runs'):
    """
    Log per-step RMSE/MAE/W2 curves as native TensorBoard scalars in a separate run directory.
    """
    rmse = list(metrics.get('rmse_series_mean') or [])
    mae  = list(metrics.get('mae_series_mean')  or [])
    w2   = list(metrics.get('w2_series_mean')   or [])
    min_len = min(len(rmse), len(mae), len(w2))
    if min_len == 0:
        return

    safe_name = "".join(c for c in str(run_name) if c.isalnum() or c in "-_.") or "eval"
    run_dir = os.path.join(logdir, run_prefix, safe_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = tf.summary.create_file_writer(run_dir)
    with writer.as_default():
        for i in range(min_len):
            if not (np.isfinite(rmse[i]) and np.isfinite(mae[i]) and np.isfinite(w2[i])):
                continue
            tf.summary.scalar('curves/rmse', float(rmse[i]), step=i)
            tf.summary.scalar('curves/mae',  float(mae[i]),  step=i)
            tf.summary.scalar('curves/w2',   float(w2[i]),   step=i)
    writer.flush()
    writer.close()


# --- Pretty-print evaluation metrics in a multi-line compact format
def print_eval_metrics(metrics: dict, header: str = "Eval", step: int | None = None, show_pos_fracs: bool = True, width: int = 80) -> None:
    """
    Pretty-print evaluation metrics in a consistent, compact multi-line format.
    Uses tqdm.write to avoid clobbering progress bars.

    Args:
        metrics: dict returned by compute_eval.
        header: label for the section (e.g., 'Eval', 'Heuristic Eval').
        step: optional outer training step for display.
        show_pos_fracs: whether to include PosImprove% summary line.
        width: width of the dashed separators.
    """
    sep = "-" * max(20, width)
    head = f"[{header} @ {step}]" if step is not None else f"[{header}]"
    rmse_summary = summarize_metric_series(metrics.get('rmse_series_mean') or [])
    mae_summary = summarize_metric_series(metrics.get('mae_series_mean') or [])
    w2_summary = summarize_metric_series(metrics.get('w2_series_mean') or [])

    tqdm.write(sep)
    tqdm.write(head)
    tqdm.write(
        f"MAE  Δ={mae_summary['delta']:.4f} %={mae_summary['relative_improvement']:.2%} "
        f"(init {mae_summary['initial']:.4f} → final {mae_summary['final']:.4f})"
    )
    tqdm.write(
        f"RMSE Δ={rmse_summary['delta']:.4f} %={rmse_summary['relative_improvement']:.2%} "
        f"(init {rmse_summary['initial']:.4f} → final {rmse_summary['final']:.4f})"
    )
    tqdm.write(
        f"W2   Δ={w2_summary['delta']:.4f} %={w2_summary['relative_improvement']:.2%} "
        f"(init {w2_summary['initial']:.4f} → final {w2_summary['final']:.4f})"
    )
    if show_pos_fracs:
        tqdm.write(
            f"PosImprove% — RMSE {rmse_summary['pos_improve_frac']:.2%} | "
            f"MAE {mae_summary['pos_improve_frac']:.2%} | "
            f"W2 {w2_summary['pos_improve_frac']:.2%}"
        )
    tqdm.write(sep)


__all__ = [
    'compute_eval',
    'heightmap_pointcloud',
    'normalized_auc',
    'wasserstein_distance_2d',
    'EMD_MAX_POINTS',
    'WASSERSTEIN_MAX_POINTS',
    'chamfer_distance',
    'earth_movers_distance',
    'log_eval_metric_curves',
    'print_eval_metrics',
    'summarize_metric_series',
]
