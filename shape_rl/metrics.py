"""Evaluation metrics for sand shaping."""

from __future__ import annotations

import os
import tensorflow as tf
from typing import Callable, Dict, Any, Sequence
from contextlib import suppress

import numpy as np
from scipy.spatial.distance import cdist
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_policy as _py_policy

from tqdm.auto import tqdm

from shape_rl.terrain import HeightMap


WASSERSTEIN_MAX_POINTS = 1024


def normalized_auc(series: list[float] | np.ndarray) -> float:
    s = np.asarray(series, dtype=np.float64)
    if s.size <= 1:
        return 0.0
    # Scale by total steps and initial magnitude to reflect per-step mean area
    denom = max(s.size, 1) * max(abs(float(s[0])), 1e-12)
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


def compute_eval(env_factory: Callable[[int | None], Any], policy, num_episodes: int = 10,
                 base_seed: int | None = None,
                 w2_stride: int = 4, w2_reg: float = 0.05,
                 w2_max_points: int = WASSERSTEIN_MAX_POINTS) -> Dict[str, Any]:
    if num_episodes is None or num_episodes <= 0:
        return {
            'rmse_series_list': [],
            'mae_series_list': [],
            'w2_series_list': [],
            'reward_series_list': [],
            'steps_series_list': [],
        }

    rmse_series_list: list[list[float]] = []
    mae_series_list: list[list[float]] = []
    w2_series_list: list[list[float]] = []
    steps_list: list[int] = []
    reward_series_list: list[list[float]] = []

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
            w20 = wasserstein_distance_2d(diff0,
                                          target_map=None,
                                          stride=w2_stride, reg=w2_reg,
                                          max_points=w2_max_points)

            rmse_series = [rmse0]
            mae_series = [mae0]
            w2_series = [w20]
            reward_series = [0.0]
            steps = 0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                action = action_step.action
                if is_py_policy:
                    time_step = env.step(action)
                else:
                    time_step = tf_env.step(action)

                steps += 1
                diff = env._env_map.difference(env._target_map)
                rmse_series.append(float(np.sqrt(np.mean(diff ** 2))))
                mae_series.append(float(np.mean(np.abs(diff))))
                w2_series.append(
                    wasserstein_distance_2d(
                        diff,
                        target_map=None,
                        stride=w2_stride,
                        reg=w2_reg,
                        max_points=w2_max_points,
                    )
                )
                try:
                    reward_series.append(float(np.mean(time_step.reward)))
                except Exception:
                    reward_series.append(np.nan)
        finally:
            with suppress(Exception):
                (tf_env.close() if tf_env is not None else None)
            with suppress(Exception):
                env.close()

        rmse_series_list.append(rmse_series)
        mae_series_list.append(mae_series)
        w2_series_list.append(w2_series)
        steps_list.append(int(steps))
        reward_series_list.append(reward_series)

    return {
        'rmse_series_list': rmse_series_list,
        'mae_series_list': mae_series_list,
        'w2_series_list': w2_series_list,
        'steps_series_list': steps_list,
        'reward_series_list': reward_series_list,
    }


def summarize_metric_series(series_or_list: Sequence[float] | Sequence[Sequence[float]] | None,
                            pos_tol: float = 1e-9) -> Dict[str, float]:
    """
    Summarize a single per-step metric series or a list of series.
    Uses step-weighted aggregation across episodes so longer episodes contribute proportionally.
    """
    fields = ('initial', 'final', 'delta', 'slope', 'auc_norm', 'pos_improve_frac', 'relative_improvement')
    fallback = {key: -1.0 for key in fields}

    def _summarize_one(series: Sequence[float]) -> tuple[Dict[str, float], int, int]:
        arr = np.asarray(series, dtype=np.float64)
        if arr.size == 0:
            return fallback, 0, 0
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return fallback, 0, 0

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
            return fallback, 0, 0

        scale = max(abs(float(values[0])), 1e-6)
        steps = start_idx + np.arange(values.size, dtype=np.float64)
        initial = float(values[0])
        final = float(values[-1])
        raw_delta = initial - final
        delta = raw_delta / scale
        slope_raw = float(np.polyfit(steps, values, 1)[0]) if values.size >= 2 else 0.0
        slope = slope_raw / scale
        auc_norm = normalized_auc(values.tolist())
        diffs = np.diff(values)
        improve_count = int(np.sum(diffs < -pos_tol)) if diffs.size > 0 else 0
        total_count = int(diffs.size)
        pos_improve_frac = float(improve_count / max(total_count, 1))
        rel_improvement = raw_delta / scale
        summary = {
            'initial': initial,
            'final': final,
            'delta': delta,
            'slope': slope,
            'auc_norm': auc_norm,
            'pos_improve_frac': pos_improve_frac,
            'relative_improvement': rel_improvement,
        }
        return summary, improve_count, total_count

    if series_or_list is None:
        return fallback

    is_series_list = (
        isinstance(series_or_list, Sequence)
        and len(series_or_list) > 0
        and isinstance(series_or_list[0], Sequence)
        and not np.isscalar(series_or_list[0])
    )

    if not is_series_list:
        summary, _, _ = _summarize_one(series_or_list)  # type: ignore[arg-type]
        return summary

    acc_weighted: dict[str, float] = {key: 0.0 for key in fields}
    improve_total = 0
    count_total = 0
    total_weight = 0.0

    for series in series_or_list:  # type: ignore[assignment]
        summary, improve_count, total_count = _summarize_one(series)
        weight = float(max(total_count, 1))
        total_weight += weight
        improve_total += improve_count
        count_total += total_count
        for key in fields:
            val = summary.get(key, -1.0)
            if np.isfinite(val):
                acc_weighted[key] += val * weight

    aggregated = {}
    for key in fields:
        aggregated[key] = (acc_weighted[key] / total_weight) if total_weight > 0 else -1.0

    aggregated['pos_improve_frac'] = (float(improve_total) / float(max(count_total, 1))) if count_total > 0 else -1.0
    return aggregated


# --- Helper: log per-step RMSE/MAE/W2 curves as native TensorBoard scalars in a separate run ---
def log_eval_metric_curves(metrics: dict, logdir: str, run_name: str, run_prefix: str = 'eval_runs'):
    """
    Log per-step RMSE/MAE/W2 curves as native TensorBoard scalars in a separate run directory.
    Prefer the first episode's curves when per-episode series are available.
    """
    rmse_list = metrics.get('rmse_series_list')
    mae_list = metrics.get('mae_series_list')
    w2_list = metrics.get('w2_series_list')
    reward_series = metrics.get('reward_series_list')

    if not (rmse_list and mae_list and w2_list):
        return

    rmse = list(np.asarray(rmse_list[0], dtype=np.float64))
    mae = list(np.asarray(mae_list[0], dtype=np.float64))
    w2 = list(np.asarray(w2_list[0], dtype=np.float64))
    reward = list(np.asarray(reward_series[0], dtype=np.float64))

    min_len = min(len(rmse), len(mae), len(w2), len(reward))
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
            tf.summary.scalar('curves/_rmse', float(rmse[i]), step=i)
            tf.summary.scalar('curves/_mae',  float(mae[i]),  step=i)
            tf.summary.scalar('curves/_w2',   float(w2[i]),   step=i)
            tf.summary.scalar('curves/reward_return', float(reward[i]), step=i)
    writer.flush()
    writer.close()


# --- Pretty-print evaluation metrics in a multi-line compact format
def print_eval_metrics(metrics: dict, header: str = "Eval", step: int | None = None, width: int = 80) -> None:
    """
    Pretty-print evaluation metrics in a consistent, compact multi-line format.
    Uses tqdm.write to avoid clobbering progress bars.

    Args:
        metrics: dict returned by compute_eval.
        header: label for the section (e.g., 'Eval', 'Heuristic Eval').
        step: optional outer training step for display.
        width: width of the dashed separators.
    """
    sep = "-" * max(20, width)
    head = f"[{header} @ {step}]" if step is not None else f"[{header}]"
    rmse_summary = summarize_metric_series(metrics.get('rmse_series_list'))
    mae_summary = summarize_metric_series(metrics.get('mae_series_list'))
    w2_summary = summarize_metric_series(metrics.get('w2_series_list'))
    steps_series = metrics.get('steps_series_list')
    steps_mean_val = None
    try:
        if steps_series is not None:
            steps_mean_val = int(round(float(np.mean(steps_series))))
    except Exception:
        steps_mean_val = None

    reward_series = metrics.get('reward_series_list') or []
    episode_returns: list[float] = []
    for curve in reward_series:
        if not curve:
            continue
        with suppress(Exception):
            episode_returns.append(float(np.nanmean(curve)))

    return_rmse_corr = np.nan
    try:
        if episode_returns and len(episode_returns) > 1:
            rmse_improve_list = []
            for series in metrics.get('rmse_series_list') or []:
                rmse_improve_list.append(summarize_metric_series(series).get('relative_improvement', np.nan))
            ep_returns_arr = np.asarray(episode_returns, dtype=np.float64)
            rmse_improve_arr = np.asarray(rmse_improve_list, dtype=np.float64)
            mask = np.isfinite(ep_returns_arr) & np.isfinite(rmse_improve_arr)
            if np.count_nonzero(mask) > 1:
                return_rmse_corr = float(np.corrcoef(ep_returns_arr[mask], rmse_improve_arr[mask])[0, 1])
    except Exception:
        return_rmse_corr = np.nan

    def _format_rel_improve(summary: dict) -> str:
        rel = summary.get('relative_improvement', -1.0)
        if not np.isfinite(rel) or rel < -0.999:
            return "n/a"
        final_norm = 1.0 - rel
        return f"1.00 → {final_norm:.2f} ({rel:.2%})"

    tqdm.write(sep)
    tqdm.write(head)
    if steps_mean_val is not None:
        tqdm.write(f"Episode Steps: {steps_mean_val:d}")
    tqdm.write(
        "Δ% — RMSE {rmse} | MAE {mae} | W2 {w2}".format(
            rmse=_format_rel_improve(rmse_summary),
            mae=_format_rel_improve(mae_summary),
            w2=_format_rel_improve(w2_summary),
        )
    )
    tqdm.write(
        f"PosImprove% — RMSE {rmse_summary['pos_improve_frac']:.2%} | "
        f"MAE {mae_summary['pos_improve_frac']:.2%} | "
        f"W2 {w2_summary['pos_improve_frac']:.2%}"
    )
    if episode_returns:
        ret_mean = float(np.mean(episode_returns))
        corr_str = "n/a"
        if np.isfinite(return_rmse_corr):
            corr_str = f"{return_rmse_corr:.3f}"
        tqdm.write(f"Return — mean {ret_mean:.3f} | corr(return, ΔRMSE) {corr_str}")
    if reward_series:
        first_curve = reward_series[0]
        if first_curve:
            try:
                tqdm.write(f"First-episode return curve samples: start {first_curve[0]:.3f} end {first_curve[-1]:.3f}")
            except Exception:
                pass
    tqdm.write(sep)


__all__ = [
    'compute_eval',
    'normalized_auc',
    'wasserstein_distance_2d',
    'WASSERSTEIN_MAX_POINTS',
    'log_eval_metric_curves',
    'print_eval_metrics',
    'summarize_metric_series',
]
