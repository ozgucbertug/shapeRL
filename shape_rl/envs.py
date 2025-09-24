"""TF-Agents compatible sand shaping environment."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .terrain import HeightMap
from .rewards import RewardInputs, RewardParams, compute_press_reward


class SandShapingEnv(py_environment.PyEnvironment):
    """Sand shaping task environment with shaped rewards."""

    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        patch_width: int = 128,
        patch_height: int = 128,
        scale_range=(1, 2),
        target_scale_range=(2, 4),
        amplitude_range=(10.0, 40.0),
        tool_radius: int = 9,
        max_steps: int = 200,
        max_push_mult: float = 1.0,
        error_threshold: float = 0.05,
        success_bonus: float = 1.0,
        fail_penalty: float = -1.0,
        terminate_on_success: bool = True,
        fail_on_breach: bool = True,
        volume_penalty_coeff: float = 3e-4,
        no_touch_penalty: float = 0.05,
        debug: bool = False,
        seed: int | None = None,
    ):
        self.debug = debug
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._width = width
        self._height = height
        self._patch_width = patch_width
        self._patch_height = patch_height
        self._scale_range = scale_range
        self._target_scale_range = target_scale_range
        self._amplitude_range = amplitude_range
        self._amp_max = self._amplitude_range[1]
        self._inv_scale_d = 2.0 / self._amp_max
        self._inv_scale_h = 1.0 / self._amp_max

        self._tool_radius = tool_radius
        self._max_steps = max_steps
        self.max_push_mult = max_push_mult

        self._error_threshold = error_threshold
        scale = max(self._inv_scale_d, 1e-6)
        self._error_threshold_abs = error_threshold / scale
        self._success_bonus = success_bonus
        self._fail_penalty = fail_penalty
        self._terminate_on_success = terminate_on_success
        self._fail_on_breach = fail_on_breach

        self._volume_penalty_coeff = volume_penalty_coeff
        self._no_touch_penalty = no_touch_penalty
        self._eps = 1e-6
        self._progress_bonus = 0.05
        self._best_err = np.inf

        self._efficiency_coeff = 0.10
        self._waste_penalty_coeff = 0.05
        self._local_radius = 3 * self._tool_radius
        self._grad_radius = 2 * self._tool_radius
        self._tool_area = np.pi * (self._tool_radius ** 2)
        self._max_press_volume = self._tool_area * (self.max_push_mult * self._tool_radius)
        self._grad_weight = 0.2
        self._lap_weight = 0.1

        self._work_diff = np.empty((self._patch_height, self._patch_width), dtype=np.float32)
        self._env_norm_buf = np.empty_like(self._work_diff)
        self._tgt_norm_buf = np.empty_like(self._work_diff)
        self._grad_buf = np.empty_like(self._work_diff)
        self._lap_buf = np.empty_like(self._work_diff)
        self._obs_buf = np.empty((self._patch_height, self._patch_width, 5), dtype=np.float32)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(4,),
            dtype=np.float32,
            minimum=np.array([0, 0, 0, 0], dtype=np.float32),
            maximum=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._patch_height, self._patch_width, 5),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='observation'
        )

        self._episode_ended = False
        self._env_map: HeightMap | None = None
        self._target_map: HeightMap | None = None
        self._tgt_mean = 0.0
        self._last_reward_terms: dict[str, float] = {}
        self._reward_params: RewardParams | None = None

        self.reset()
        assert self._env_map is not None
        assert self._target_map is not None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _local_rmse(self, diff_map, cx, cy, r):
        y0 = max(cy - r, 0)
        y1 = min(cy + r + 1, self._height)
        x0 = max(cx - r, 0)
        x1 = min(cx + r + 1, self._width)
        patch = diff_map[y0:y1, x0:x1]
        return float(np.sqrt(np.mean(patch ** 2)))

    def _diff_and_rmse(self):
        assert self._env_map is not None
        assert self._target_map is not None
        diff = self._env_map.difference(self._target_map, out=self._work_diff)
        err = float(np.sqrt(np.mean(diff ** 2)))
        return diff, err

    def _laplacian(self, diff):
        lap = self._lap_buf
        lap.fill(0.0)
        lap += np.roll(diff, 1, axis=0)
        lap += np.roll(diff, -1, axis=0)
        lap += np.roll(diff, 1, axis=1)
        lap += np.roll(diff, -1, axis=1)
        lap -= 4.0 * diff
        return lap

    def _create_carved_target(self, base_map: HeightMap) -> HeightMap:
        carved = base_map.clone()
        max_depth = self.max_push_mult * self._tool_radius
        if max_depth <= 0:
            return carved

        num_clusters = int(self._rng.integers(4, 8))
        for _ in range(num_clusters):
            cx = float(self._rng.uniform(self._tool_radius, self._patch_width - self._tool_radius - 1))
            cy = float(self._rng.uniform(self._tool_radius, self._patch_height - self._tool_radius - 1))
            base_depth = float(self._rng.uniform(0.4, 1.0) * max_depth)
            streak_len = int(self._rng.integers(2, 5))
            angle = float(self._rng.uniform(0.0, 2 * np.pi))
            step = float(self._rng.uniform(0.25, 0.75) * self._tool_radius)
            for k in range(streak_len):
                jitter_x = cx + k * step * np.cos(angle) + self._rng.normal(0.0, 0.3 * self._tool_radius)
                jitter_y = cy + k * step * np.sin(angle) + self._rng.normal(0.0, 0.3 * self._tool_radius)
                jitter_x = float(np.clip(jitter_x, 0.0, self._patch_width - 1))
                jitter_y = float(np.clip(jitter_y, 0.0, self._patch_height - 1))
                depth_scale = float(self._rng.uniform(0.6, 1.1))
                depth = np.clip(base_depth * depth_scale, 0.05, max_depth)
                map_view = carved.map
                cy_i = int(np.clip(round(jitter_y), 0, self._patch_height - 1))
                cx_i = int(np.clip(round(jitter_x), 0, self._patch_width - 1))
                z_abs = float(map_view[cy_i, cx_i])
                carved.apply_press_abs(jitter_x, jitter_y, z_abs, depth)
        return carved

    def _build_observation(self, diff, h, t):
        assert self._env_map is not None
        np.multiply(diff, self._inv_scale_d, out=self._work_diff)
        np.clip(self._work_diff, -1.0, 1.0, out=self._work_diff)

        np.subtract(h, self._env_map._mean, out=self._env_norm_buf)
        self._env_norm_buf *= self._inv_scale_h
        np.clip(self._env_norm_buf, -1.0, 1.0, out=self._env_norm_buf)

        np.subtract(t, self._tgt_mean, out=self._tgt_norm_buf)
        self._tgt_norm_buf *= self._inv_scale_h
        np.clip(self._tgt_norm_buf, -1.0, 1.0, out=self._tgt_norm_buf)

        gy, gx = np.gradient(diff)
        np.sqrt(gy * gy + gx * gx, out=self._grad_buf)
        np.multiply(self._grad_buf, self._inv_scale_d, out=self._grad_buf)
        np.clip(self._grad_buf, -1.0, 1.0, out=self._grad_buf)

        lap = self._laplacian(diff)
        np.multiply(lap, self._inv_scale_d, out=self._lap_buf)
        np.clip(self._lap_buf, -1.0, 1.0, out=self._lap_buf)

        self._obs_buf[..., 0] = self._work_diff
        self._obs_buf[..., 1] = self._env_norm_buf
        self._obs_buf[..., 2] = self._tgt_norm_buf
        self._obs_buf[..., 3] = self._grad_buf
        self._obs_buf[..., 4] = self._lap_buf
        return self._obs_buf

    def _reset(self):
        scale_x = self._rng.uniform(self._scale_range[0], self._scale_range[1])
        scale_y = self._rng.uniform(self._scale_range[0], self._scale_range[1])
        amplitude = self._rng.uniform(self._amplitude_range[0], self._amplitude_range[1])
        env_seed = None if self._seed is None else int(self._rng.integers(0, 2**32))
        self._env_map = HeightMap(
            self._patch_width,
            self._patch_height,
            scale=(scale_x, scale_y),
            amplitude=amplitude,
            tool_radius=self._tool_radius,
            seed=env_seed,
            bedrock_offset=30,
        )
        self._target_map = self._create_carved_target(self._env_map)
        self._tgt_mean = self._target_map._mean

        self._step_count = 0
        self._episode_ended = False

        diff = self._env_map.difference(self._target_map, out=self._work_diff)
        h = self._env_map.map
        t = self._target_map.map
        obs = self._build_observation(diff, h, t)

        self._err0 = float(np.sqrt(np.mean(diff ** 2))) + self._eps
        self._best_err = self._err0
        self._reward_params = RewardParams(
            eps=self._eps,
            err0=self._err0,
            best_err=self._best_err,
            max_press_volume=self._max_press_volume,
            no_touch_penalty=self._no_touch_penalty,
            volume_penalty_coeff=self._volume_penalty_coeff,
            progress_bonus_scale=self._progress_bonus,
        )

        if self.debug:
            self._initial_err0 = self._err0

        return ts.restart(obs)

    def _step(self, action: np.ndarray):
        if self._episode_ended:
            return self._reset()

        assert self._env_map is not None
        assert self._target_map is not None
        assert self._reward_params is not None

        x_norm, y_norm, dz_norm, smooth_norm = action
        x = x_norm * (self._patch_width - 1)
        y = y_norm * (self._patch_height - 1)

        cy = int(np.clip(round(y), 0, self._patch_height - 1))
        cx = int(np.clip(round(x), 0, self._patch_width - 1))
        h_center = float(self._env_map.map[cy, cx])
        z_abs = h_center
        dz_rel = dz_norm * (self.max_push_mult * self._tool_radius)

        diff_before, err_g_before = self._diff_and_rmse()
        err_l_before = self._local_rmse(diff_before, cx, cy, self._local_radius)
        gy_b, gx_b = np.gradient(diff_before)
        grad_before = float(np.sqrt(np.mean(gy_b * gy_b + gx_b * gx_b)))
        lap_before_arr = self._laplacian(diff_before).copy()
        lap_before = float(np.sqrt(np.mean(lap_before_arr ** 2)))

        removed, touched = self._env_map.apply_press_abs(x, y, z_abs, dz_rel)
        if smooth_norm > 1e-3:
            self._env_map.smooth_patch(x, y, smooth_norm)

        diff_after, err_g_after = self._diff_and_rmse()
        err_l_after = self._local_rmse(diff_after, cx, cy, self._local_radius)
        gy_a, gx_a = np.gradient(diff_after)
        grad_after = float(np.sqrt(np.mean(gy_a * gy_a + gx_a * gx_a)))
        lap_after_arr = self._laplacian(diff_after)
        lap_after = float(np.sqrt(np.mean(lap_after_arr ** 2)))

        reward_inputs = RewardInputs(
            err_g_before=err_g_before,
            err_g_after=err_g_after,
            err_l_before=err_l_before,
            err_l_after=err_l_after,
            removed=removed,
            touched=touched,
            grad_before=grad_before,
            grad_after=grad_after,
            lap_before=lap_before,
            lap_after=lap_after,
            smooth_strength=float(smooth_norm),
        )
        result = compute_press_reward(self._reward_params, reward_inputs)
        reward = result.reward
        self._best_err = result.best_err
        self._reward_params = replace(self._reward_params, best_err=self._best_err)

        if self.debug:
            self._last_removed_norm = removed / (self._max_press_volume + self._eps)
            self._last_rel_improve = (err_g_before - err_g_after) / (err_g_before + self._eps)
            self._last_reward = reward
            self._last_touched = touched
            self._last_err_global = err_g_after
            self._last_err_local = err_l_after
            self._last_removed = removed
            self._last_grad = grad_after
            self._last_lap = lap_after
            self._last_reward_terms = result.diagnostics

        if self._terminate_on_success and err_g_after <= self._error_threshold_abs:
            self._episode_ended = True
            reward += self._success_bonus
            h = self._env_map.map
            t = self._target_map.map
            obs = self._build_observation(diff_after, h, t)
            return ts.termination(obs, reward)

        if self._fail_on_breach and np.any(self._env_map.map <= (self._env_map.bedrock + self._eps)):
            self._episode_ended = True
            reward += self._fail_penalty
            h = self._env_map.map
            t = self._target_map.map
            obs = self._build_observation(diff_after, h, t)
            return ts.termination(obs, reward)

        self._step_count += 1

        h = self._env_map.map
        t = self._target_map.map
        obs = self._build_observation(diff_after, h, t)

        if self._step_count >= self._max_steps:
            self._episode_ended = True
            return ts.termination(obs, reward)

        return ts.transition(obs, reward, discount=1.0)


__all__ = ['SandShapingEnv']
