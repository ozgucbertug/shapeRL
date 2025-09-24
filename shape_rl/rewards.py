"""Reward computation utilities for sand shaping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class RewardParams:
    """Static per-environment reward parameters."""
    eps: float
    err0: float
    best_err: float
    best_chamfer: float
    best_emd: float
    max_press_volume: float
    no_touch_penalty: float
    volume_penalty_coeff: float
    progress_bonus_scale: float
    touch_bonus: float
    idle_penalty: float
    smooth_grad_coeff: float
    smooth_cost_coeff: float
    milestone_threshold: float
    milestone_bonus: float


@dataclass
class RewardInputs:
    err_g_before: float
    err_g_after: float
    err_l_before: float
    err_l_after: float
    removed: float
    touched: bool
    grad_before: float
    grad_after: float
    lap_before: float
    lap_after: float
    smooth_strength: float
    chamfer_before: float
    chamfer_after: float
    emd_before: float
    emd_after: float


@dataclass
class RewardResult:
    reward: float
    best_err: float
    best_chamfer: float
    best_emd: float
    diagnostics: Dict[str, float]


def compute_press_reward(params: RewardParams, inputs: RewardInputs) -> RewardResult:
    """Compute shaped reward for a single press action."""
    delta_g = inputs.err_g_before - inputs.err_g_after
    delta_l = inputs.err_l_before - inputs.err_l_after

    norm_g = max(params.err0, inputs.err_g_before, params.eps)
    norm_l = max(params.err0 * 0.5, inputs.err_l_before, params.eps)

    rel_g = delta_g / norm_g
    rel_l = delta_l / norm_l

    progress = 0.75 * rel_g + 0.25 * rel_l
    progress_score = np.tanh(progress * 4.0)

    vol_norm = inputs.removed / (params.max_press_volume + params.eps)
    reward = progress_score
    touch_bonus = params.touch_bonus if inputs.touched else -params.no_touch_penalty
    reward += touch_bonus

    overdig = 0.0
    if inputs.touched:
        expected = max(progress_score, 0.0)
        overdig = max(0.0, vol_norm - expected)
        reward -= params.volume_penalty_coeff * overdig * params.max_press_volume
    else:
        reward -= params.idle_penalty

    regression_penalty = 0.1 * max(0.0, -progress_score)
    reward -= regression_penalty

    if inputs.smooth_strength > 1e-3 and inputs.touched:
        grad_drop = max(0.0, inputs.grad_before - inputs.grad_after)
        reward += params.smooth_grad_coeff * grad_drop
        reward -= params.smooth_cost_coeff * inputs.smooth_strength

    progress_bonus = 0.0
    best_err = params.best_err
    if inputs.err_g_after < params.best_err - 1e-6:
        improvement = (params.best_err - inputs.err_g_after) / (params.err0 + params.eps)
        progress_bonus = params.progress_bonus_scale * (1.0 + np.clip(improvement, 0.0, 1.0))
        reward += progress_bonus
        best_err = inputs.err_g_after

    milestone_bonus = 0.0
    best_chamfer = params.best_chamfer
    best_emd = params.best_emd
    chamfer_improvement = (inputs.chamfer_before - inputs.chamfer_after) / max(inputs.chamfer_before, params.eps)
    emd_improvement = (inputs.emd_before - inputs.emd_after) / max(inputs.emd_before, params.eps)
    if chamfer_improvement > params.milestone_threshold and inputs.chamfer_after < best_chamfer:
        milestone_bonus += params.milestone_bonus * chamfer_improvement
        best_chamfer = inputs.chamfer_after
    if emd_improvement > params.milestone_threshold and inputs.emd_after < best_emd:
        milestone_bonus += params.milestone_bonus * emd_improvement
        best_emd = inputs.emd_after
    reward += milestone_bonus

    reward_clipped = float(np.clip(reward, -3.0, 3.0))

    diagnostics = {
        'delta_g': float(delta_g),
        'delta_l': float(delta_l),
        'rel_g': float(rel_g),
        'rel_l': float(rel_l),
        'vol_norm': float(vol_norm),
        'overdig': float(overdig),
        'touch_bonus': float(touch_bonus),
        'regression_penalty': float(regression_penalty),
        'progress_score': float(progress_score),
        'progress_bonus': float(progress_bonus),
        'milestone_bonus': float(milestone_bonus),
        'reward_raw': float(reward),
        'reward_clipped': reward_clipped,
    }

    return RewardResult(
        reward=reward_clipped,
        best_err=best_err,
        best_chamfer=best_chamfer,
        best_emd=best_emd,
        diagnostics=diagnostics,
    )


__all__ = [
    'RewardParams',
    'RewardInputs',
    'RewardResult',
    'compute_press_reward',
]
