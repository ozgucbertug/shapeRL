import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from .terrain import HeightMap


class SandShapingEnv(py_environment.PyEnvironment):
    def __init__(
        self,
        # ── GEOMETRY & PROCEDURAL MAP ──────────────────────────────
        width: int = 128,
        height: int = 128,
        patch_width: int = 128,
        patch_height: int = 128,
        scale_range=(1, 2),
        target_scale_range=(2, 4),
        amplitude_range=(10.0, 40.0),
        # ── TOOL / ACTION & EPISODE HORIZON ───────────────────────
        tool_radius: int = 9,
        max_steps: int = 200,
        max_push_mult: float = 1.0,
        # ── TERMINATION & OUTCOME BONUSES ─────────────────────────
        error_threshold: float = 0.05,
        success_bonus: float = 1.0,
        fail_penalty: float = -1.0,
        terminate_on_success: bool = True,
        fail_on_breach: bool = True,
        # ── REWARD SHAPING / SCALING KNOBS ────────────────────────
        volume_penalty_coeff: float = 3e-4,
        no_touch_penalty: float = 0.02,
        # ── MISC / DEBUG ──────────────────────────────────────────
        debug: bool = False,
        seed: int | None = None,
    ):
        self.debug = debug
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # ── GEOMETRY & PROCEDURAL MAP ──────────────────────────────
        self._width = width
        self._height = height
        self._patch_width = patch_width
        self._patch_height = patch_height
        self._scale_range = scale_range
        self._target_scale_range = target_scale_range
        self._amplitude_range = amplitude_range
        self._amp_max = self._amplitude_range[1]
        # scaling for obs
        self._inv_scale_d = 2.0 / self._amp_max   # diff/grad/lap
        self._inv_scale_h = 1.0 / self._amp_max   # heights

        # Per-episode robust scales for diff/grad/lap
        self._s_diff = 1.0
        self._s_grad = 1.0
        self._s_lap  = 1.0

        # ── TOOL / ACTION & EPISODE HORIZON ───────────────────────
        self._tool_radius = tool_radius
        self._max_steps = max_steps
        self.max_push_mult = max_push_mult

        # ── TERMINATION & OUTCOME BONUSES ─────────────────────────
        self._error_threshold      = error_threshold
        self._success_bonus        = success_bonus
        self._fail_penalty         = fail_penalty
        self._terminate_on_success = terminate_on_success
        self._fail_on_breach       = fail_on_breach

        # ── REWARD SHAPING / SCALING KNOBS ────────────────────────
        self._volume_penalty_coeff = volume_penalty_coeff
        self._step_cost            = float(no_touch_penalty)  # per-action cost to discourage dithering
        self._eps                  = 1e-6
        self._global_weight        = 0.7
        self._local_weight         = 0.3

        self._local_radius = 3 * self._tool_radius
        self._max_press_volume = self._compute_max_press_volume()
        self._last_reward_terms: dict[str, float] = {}

        # ── INTERNAL WORK BUFFERS ─────────────────────────────────
        H, W = self._patch_height, self._patch_width
        self._work_diff    = np.empty((H, W), dtype=np.float32)
        self._env_norm_buf = np.empty((H, W), dtype=np.float32)
        self._tgt_norm_buf = np.empty((H, W), dtype=np.float32)
        self._grad_buf     = np.empty((H, W), dtype=np.float32)  # |∇diff|
        self._lap_buf      = np.empty((H, W), dtype=np.float32)  # Δ diff
        self._obs_buf      = np.empty((H, W, 6), dtype=np.float32)  # [diff, env, tgt, grad, lap, progress]

        # ── SPECS ────────────────────────────────────────────────
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,),
            dtype=np.float32,
            minimum=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            maximum=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(H, W, 6),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='observation'
        )

        # ── STATE VARIABLES ──────────────────────────────────────
        self._episode_ended = False
        self._env_map = None
        self._target_map = None
        self._tgt_mean = 0.0
        self._current_err_global = np.inf

        self.reset()
        assert self._env_map is not None
        assert self._target_map is not None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _compute_max_press_volume(self) -> float:
        """Maximum excavation volume for a single press (spherical-cap geometry)."""
        radius = float(self._tool_radius)
        depth = float(np.clip(self.max_push_mult * radius, 0.0, 2.0 * radius))
        if depth <= 0.0:
            return self._eps
        return float((np.pi * depth * depth * (3.0 * radius - depth)) / 3.0)

    # ---------------------------------------------------- #
    #  Local RMSE around (cx, cy) within square radius r   #
    # ---------------------------------------------------- #
    def _local_rmse(self, diff_map, cx, cy, r):
        y0 = max(cy - r, 0)
        y1 = min(cy + r + 1, self._height)
        x0 = max(cx - r, 0)
        x1 = min(cx + r + 1, self._width)
        patch = diff_map[y0:y1, x0:x1]
        return float(np.sqrt(np.mean(patch ** 2)))

    def _diff_and_rmse(self):
        """Compute and return the current diff map and its global RMSE."""
        assert self._env_map is not None
        assert self._target_map is not None
        diff = self._env_map.difference(self._target_map, out=self._work_diff)
        err = float(np.sqrt(np.mean(diff ** 2)))
        return diff, err

    # -------------------------- #
    #  Discrete 5-point Laplace  #
    # -------------------------- #
    def _laplacian(self, diff: np.ndarray) -> np.ndarray:
        """
        Compute 5-point Laplacian of `diff` with edge padding.
        Writes into self._lap_buf and returns it.
        """
        padded = np.pad(diff, 1, mode='edge')
        center = padded[1:-1, 1:-1]
        lap = self._lap_buf
        # Δ = (N + S + E + W - 4C)
        np.subtract(padded[2:, 1:-1], center, out=lap)   # S - C
        lap += padded[:-2, 1:-1] - center                # + (N - C)
        lap += padded[1:-1, 2:] - center                 # + (E - C)
        lap += padded[1:-1, :-2] - center                # + (W - C)
        return lap

    def _update_grad_lap(self, diff: np.ndarray):
        """Update gradient magnitude and Laplacian buffers with per-episode robust scaling."""
        # Gradient magnitude (non-negative)
        gy, gx = np.gradient(diff)
        np.sqrt(gy * gy + gx * gx, out=self._grad_buf)  # |∇diff|
        scale_grad = self._s_grad if self._s_grad > self._eps else 1.0
        np.divide(self._grad_buf, scale_grad, out=self._grad_buf)
        # Clip to [0, 1] since this is a magnitude
        np.clip(self._grad_buf, 0.0, 1.0, out=self._grad_buf)

        # Signed Laplacian
        lap = self._laplacian(diff)  # writes into self._lap_buf
        scale_lap = self._s_lap if self._s_lap > self._eps else 1.0
        np.divide(lap, scale_lap, out=self._lap_buf)
        np.clip(self._lap_buf, -1.0, 1.0, out=self._lap_buf)

    # ------------------------------------------------------------------ #
    # Utility: build observation tensor                                  #
    # ------------------------------------------------------------------ #
    def _build_observation(self, diff, h, t):
        """Return the observation tensor using pre-allocated buffers."""
        assert self._env_map is not None

        # 0: signed diff (per-episode robust scaled)
        scale_diff = self._s_diff if self._s_diff > self._eps else 1.0
        np.divide(diff, scale_diff, out=self._work_diff)
        np.clip(self._work_diff, -1.0, 1.0, out=self._work_diff)

        # 1: env height normalized by running mean
        np.subtract(h, self._env_map._mean, out=self._env_norm_buf)
        self._env_norm_buf *= self._inv_scale_h
        np.clip(self._env_norm_buf, -1.0, 1.0, out=self._env_norm_buf)

        # 2: target height normalized by per-episode mean
        np.subtract(t, self._tgt_mean, out=self._tgt_norm_buf)
        self._tgt_norm_buf *= self._inv_scale_h
        np.clip(self._tgt_norm_buf, -1.0, 1.0, out=self._tgt_norm_buf)

        # 3–4: gradient magnitude & Laplacian of raw diff
        self._update_grad_lap(diff)

        # Assemble
        self._obs_buf[..., 0] = self._work_diff
        self._obs_buf[..., 1] = self._env_norm_buf
        self._obs_buf[..., 2] = self._tgt_norm_buf
        self._obs_buf[..., 3] = self._grad_buf
        self._obs_buf[..., 4] = self._lap_buf
        err_ratio = 0.0
        if np.isfinite(self._current_err_global):
            err_ratio = self._current_err_global / (self._err0 + self._eps)
        err_ratio = float(np.clip(err_ratio, 0.0, 1.0))
        progress_plane = 2.0 * err_ratio - 1.0
        self._obs_buf[..., 5].fill(progress_plane)
        return self._obs_buf

    # ------------------------------------------------------------------ #
    # Episode initialisation                                             #
    # ------------------------------------------------------------------ #
    def _reset(self):
        # Sample new substrate
        scale_x = self._rng.uniform(self._scale_range[0], self._scale_range[1])
        scale_y = self._rng.uniform(self._scale_range[0], self._scale_range[1])
        amplitude = self._rng.uniform(self._amplitude_range[0], self._amplitude_range[1])
        env_seed = None if self._seed is None else int(self._rng.integers(0, 2**32))
        self._env_map = HeightMap(self._patch_width,
                                  self._patch_height,
                                  scale=(scale_x, scale_y),
                                  amplitude=amplitude,
                                  tool_radius=self._tool_radius,
                                  seed=env_seed,
                                  bedrock_offset=30)

        # Sample new target
        tgt_scale_x = self._rng.uniform(self._target_scale_range[0], self._target_scale_range[1])
        tgt_scale_y = self._rng.uniform(self._target_scale_range[0], self._target_scale_range[1])
        tgt_amplitude = self._rng.uniform(self._amplitude_range[0], self._amplitude_range[1])
        tgt_seed = None if self._seed is None else int(self._rng.integers(0, 2**32))
        self._target_map = HeightMap(self._patch_width,
                                     self._patch_height,
                                     scale=(tgt_scale_x, tgt_scale_y),
                                     amplitude=tgt_amplitude,
                                     tool_radius=self._tool_radius,
                                     seed=tgt_seed)

        self._tgt_mean = self._target_map._mean
        self._step_count = 0
        self._episode_ended = False

        diff = self._env_map.difference(self._target_map, out=self._work_diff)
        h = self._env_map.map
        t = self._target_map.map

        # --- Per-episode robust scales for diff/grad/lap (95th percentile) ---
        abs_diff = np.abs(diff)
        self._s_diff = float(max(self._eps, np.percentile(abs_diff, 95.0)))

        gy, gx = np.gradient(diff)
        grad_mag = np.sqrt(gy * gy + gx * gx)
        self._s_grad = float(max(self._eps, np.percentile(grad_mag, 95.0)))

        lap = self._laplacian(diff)  # writes into self._lap_buf
        abs_lap = np.abs(lap)
        self._s_lap = float(max(self._eps, np.percentile(abs_lap, 95.0)))

        # Initial global error
        self._err0 = float(np.sqrt(np.mean(diff**2))) + self._eps
        self._current_err_global = self._err0

        # Build first observation using the new per-episode scales
        obs = self._build_observation(diff, h, t)
        if self.debug:
            self._initial_err0 = self._err0

        return ts.restart(obs)

    # ------------------------------------------------------------------ #
    # One environment step                                               #
    # ------------------------------------------------------------------ #
    def _step(self, action: np.ndarray):
        if self._episode_ended:
            return self._reset()

        assert self._env_map is not None
        assert self._target_map is not None

        # Parse action
        x_norm: float = action[0]
        y_norm: float = action[1]
        dz_norm: float = action[2]

        # Map normalized actions directly across the playable patch
        x = x_norm * (self._patch_width - 1)
        y = y_norm * (self._patch_height - 1)

        # Derive absolute tip height from current surface, then push down by dz_norm
        cy = int(np.clip(round(y), 0, self._patch_height - 1))
        cx = int(np.clip(round(x), 0, self._patch_width - 1))
        h_center = float(self._env_map.map[cy, cx])
        z_abs = h_center
        dz_rel = dz_norm * (self.max_push_mult * self._tool_radius)

        # Global & local RMSE before the press
        diff_before, err_g_before = self._diff_and_rmse()
        err_l_before = self._local_rmse(diff_before, cx, cy, self._local_radius)

        # Apply press and measure removed volume
        removed, touched = self._env_map.apply_press_abs(x, y, z_abs, dz_rel)

        # Global & local RMSE after the press
        diff_after, err_g_after = self._diff_and_rmse()
        err_l_after  = self._local_rmse(diff_after, cx, cy, self._local_radius)
        reward = self._compute_reward(
            err_g_before, err_g_after,
            err_l_before, err_l_after,
            removed
        )

        # ----- early-termination checks -----
        if self._terminate_on_success and err_g_after <= self._error_threshold:
            self._episode_ended = True
            reward += self._success_bonus
            h = self._env_map.map; t = self._target_map.map
            self._current_err_global = err_g_after
            obs = self._build_observation(diff_after, h, t)
            if self.debug:
                self._last_grad = float(np.sqrt(np.mean(self._grad_buf**2)))
                self._last_lap  = float(np.sqrt(np.mean(self._lap_buf**2)))
            return ts.termination(obs, reward)

        # Use explicit bedrock from HeightMap for breach check
        bedrock = getattr(self._env_map, "bedrock", 0.0)
        if self._fail_on_breach and np.any(self._env_map.map < bedrock - 1e-9):
            self._episode_ended = True
            reward += self._fail_penalty
            h = self._env_map.map; t = self._target_map.map
            self._current_err_global = err_g_after
            obs = self._build_observation(diff_after, h, t)
            if self.debug:
                self._last_grad = float(np.sqrt(np.mean(self._grad_buf**2)))
                self._last_lap  = float(np.sqrt(np.mean(self._lap_buf**2)))
            return ts.termination(obs, reward)

        self._step_count += 1

        h = self._env_map.map; t = self._target_map.map
        self._current_err_global = err_g_after
        obs = self._build_observation(diff_after, h, t)

        # Debug scalars used by training.py if present
        if self.debug:
            self._last_removed_norm = removed / (self._max_press_volume + self._eps)
            self._last_removed = removed
            self._last_rel_improve = (err_g_before - err_g_after) / (err_g_before + self._eps)
            self._last_reward = reward
            self._last_err_global = err_g_after
            self._last_err_local = err_l_after
            self._last_grad = float(np.sqrt(np.mean(self._grad_buf**2)))
            self._last_lap  = float(np.sqrt(np.mean(self._lap_buf**2)))

        if self._step_count >= self._max_steps:
            self._episode_ended = True
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward, discount=1.0)

    # ---------------------------------------------------- #
    #  Reward (potential-based shaping)                    #
    # ---------------------------------------------------- #
    def _compute_reward(self,
                        err_g_before: float, err_g_after: float,
                        err_l_before: float, err_l_after: float,
                        removed: float) -> float:
        # --- Relative improvements (scale-free) ---
        rel_g = (err_g_before - err_g_after) / max(err_g_before, self._eps)
        rel_l = (err_l_before - err_l_after) / max(err_l_before, self._eps)
        rel_g = float(np.clip(rel_g, -1.0, 1.0))
        rel_l = float(np.clip(rel_l, -1.0, 1.0))
        improve = self._global_weight * rel_g + self._local_weight * rel_l

        # --- Gentle volume regularizer (normalized) ---
        removed_norm = removed / (self._max_press_volume + self._eps)
        vol_pen = self._volume_penalty_coeff * removed_norm

        # --- Per-step cost (kept small/unconditional here) ---
        step_pen = self._step_cost

        reward = improve - vol_pen - step_pen

        if self.debug:
            self._last_reward_terms = {
                'rel_g': rel_g,
                'rel_l': rel_l,
                'improve': improve,
                'removed_norm': float(removed_norm),
                'vol_pen': float(vol_pen),
                'step_pen': float(step_pen),
            }
            self._last_reward = float(reward)
            self._last_err_global = float(err_g_after)
            self._last_err_local = float(err_l_after)
            self._last_rel_improve = float(improve)
        return float(reward)
