import math

import numpy as np

from shape_rl.rewards import RewardParams, RewardInputs, compute_press_reward


def test_reward_positive_on_progress():
    params = RewardParams(
        eps=1e-6,
        err0=1.0,
        best_err=1.0,
        best_chamfer=1.0,
        best_emd=1.0,
        max_press_volume=10.0,
        no_touch_penalty=0.05,
        volume_penalty_coeff=3e-4,
        progress_bonus_scale=0.05,
        touch_bonus=0.05,
        idle_penalty=0.02,
        smooth_grad_coeff=0.02,
        smooth_cost_coeff=0.01,
        milestone_threshold=0.05,
        milestone_bonus=0.05,
    )
    inputs = RewardInputs(
        err_g_before=1.0,
        err_g_after=0.4,
        err_l_before=0.8,
        err_l_after=0.3,
        removed=2.5,
        touched=True,
        grad_before=0.5,
        grad_after=0.3,
        lap_before=0.2,
        lap_after=0.1,
        smooth_strength=0.0,
        chamfer_before=1.0,
        chamfer_after=0.5,
        emd_before=1.0,
        emd_after=0.5,
    )
    result = compute_press_reward(params, inputs)
    assert result.reward > 0.0
    assert math.isfinite(result.reward)
    assert result.best_err < params.best_err
    assert 'progress_score' in result.diagnostics


def test_idle_penalty_negative_reward():
    params = RewardParams(
        eps=1e-6,
        err0=1.0,
        best_err=1.0,
        best_chamfer=1.0,
        best_emd=1.0,
        max_press_volume=10.0,
        no_touch_penalty=0.05,
        volume_penalty_coeff=3e-4,
        progress_bonus_scale=0.05,
        touch_bonus=0.05,
        idle_penalty=0.02,
        smooth_grad_coeff=0.02,
        smooth_cost_coeff=0.01,
        milestone_threshold=0.05,
        milestone_bonus=0.05,
    )
    inputs = RewardInputs(
        err_g_before=1.0,
        err_g_after=1.0,
        err_l_before=0.9,
        err_l_after=0.9,
        removed=0.0,
        touched=False,
        grad_before=0.5,
        grad_after=0.5,
        lap_before=0.1,
        lap_after=0.1,
        smooth_strength=0.0,
        chamfer_before=1.0,
        chamfer_after=1.0,
        emd_before=1.0,
        emd_after=1.0,
    )
    result = compute_press_reward(params, inputs)
    assert result.reward < 0.0
    assert math.isfinite(result.reward)
