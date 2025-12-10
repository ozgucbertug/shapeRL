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
        tool_radius: int = 5,
        max_steps: int = 1024,
        max_push_mult: float = 1.0,
        # ── TERMINATION & OUTCOME BONUSES ─────────────────────────
        error_threshold: float = 0.05,
        relative_success_frac: float = 0.05,
        plateau_min_steps: int = 64,
        plateau_patience: int = 16,
        plateau_improve_tol: float = 0.001,
        success_bonus: float = 1.0,
        fail_penalty: float = -1.0,
        terminate_on_success: bool = True,
        fail_on_breach: bool = True,
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

        # Per-episode robust scales for diff/grad/lap
        self._s_diff = 1.0
        self._s_grad = 1.0
        self._s_lap  = 1.0
        self._s_env_h = 1.0
        self._s_tgt_h = 1.0

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
        self._relative_success_frac = relative_success_frac
        self._plateau_min_steps     = plateau_min_steps
        self._plateau_patience      = plateau_patience
        self._plateau_improve_tol   = plateau_improve_tol

        # ── REWARD SHAPING / SCALING KNOBS ────────────────────────
        self._eps                  = 1e-6
        # Softer penalties and reward scale to stabilize critic targets
        self._k_deficit  = 1.5
        self._k_overcut  = 0.35
        self._alpha_over = 0.5
        self._reward_scale = 6.0
        self._vol_reg = 0.1

        # Additional reward shaping / safety weights
        self._lambda_rmse = 2.0            # weight on global RMSE improvement
        self._lambda_mae = 1.5             # weight on global MAE improvement
        self._lambda_local_surplus = 1.25  # reward for removing surplus in the pressed footprint
        self._lambda_local_grad = 0.2      # encourage local smoothing on surplus
        self._k_local_deficit = 2.5        # penalize new local deficits created by the press
        self._k_center_deficit = 2.0       # barrier if the press is commanded in a deficit
        self._overcut_vol_scale = 1.0      # scale overcut penalty by removed volume

        self._local_radius = 1 * self._tool_radius
        self._max_press_volume = self._compute_max_press_volume()
        self._depth_unit = max(self.max_push_mult * self._tool_radius, self._eps)
        self._last_reward_terms: dict[str, float] = {}
        self._success_threshold = self._error_threshold
        self._plateau_tol = 0.0

        # Precompute a circular mask for footprint-local stats
        coords = np.indices((2 * self._tool_radius + 1, 2 * self._tool_radius + 1))
        dy = coords[0] - self._tool_radius
        dx = coords[1] - self._tool_radius
        self._disk_mask = (dx * dx + dy * dy <= self._tool_radius * self._tool_radius).astype(np.float32)
        self._disk_area = float(np.sum(self._disk_mask))

        # ── INTERNAL WORK BUFFERS ─────────────────────────────────
        H, W = self._patch_height, self._patch_width
        self._work_diff    = np.empty((H, W), dtype=np.float32)
        self._env_norm_buf = np.empty((H, W), dtype=np.float32)
        self._tgt_norm_buf = np.empty((H, W), dtype=np.float32)
        self._grad_buf     = np.empty((H, W), dtype=np.float32)  # |∇diff|
        self._lap_buf      = np.empty((H, W), dtype=np.float32)  # Δ diff
        # [diff, env, tgt, grad, lap, scale_token]
        self._obs_buf      = np.empty((H, W, 6), dtype=np.float32)

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
            minimum=-5.0,
            maximum=5.0,
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

    def _surplus_rmse(self, diff_map: np.ndarray) -> float:
        """RMSE over removable surplus only (diff > 0)."""
        surplus = np.maximum(diff_map, 0.0)
        return float(np.sqrt(np.mean(surplus * surplus)))

    def _deficit_volume(self, diff_map: np.ndarray) -> float:
        """Total deficit magnitude (diff < 0)."""
        return float(np.sum(np.maximum(-diff_map, 0.0)))

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

    def _update_grad_lap(self, diff_canon: np.ndarray):
        """Update gradient magnitude and Laplacian buffers in canonical units."""
        gy, gx = np.gradient(diff_canon)
        np.sqrt(gy * gy + gx * gx, out=self._grad_buf)  # |∇diff| in canonical units per pixel
        np.clip(self._grad_buf, 0.0, 5.0, out=self._grad_buf)

        lap = self._laplacian(diff_canon)  # writes into self._lap_buf
        np.clip(self._lap_buf, -5.0, 5.0, out=self._lap_buf)

    def _disk_stats(self, diff: np.ndarray, cx: int, cy: int) -> tuple[float, float, float]:
        """Compute surplus/deficit/gradient stats inside the circular footprint."""
        r = self._tool_radius
        patch = diff[cy - r:cy + r + 1, cx - r:cx + r + 1]
        mask = self._disk_mask
        # Surplus-only structures
        surplus = np.maximum(patch, 0.0) * mask
        gy, gx = np.gradient(surplus)
        grad_l1 = float(np.mean(np.abs(gy)) + np.mean(np.abs(gx)))

        # Mean surplus/deficit within the footprint
        surplus_mae = float(np.sum(surplus) / max(self._disk_area, self._eps))
        deficit_mae = float(np.sum(np.maximum(-patch, 0.0) * mask) / max(self._disk_area, self._eps))
        return surplus_mae, deficit_mae, grad_l1

    def _gather_metrics(self, diff: np.ndarray, cx: int, cy: int, err_global: float | None = None) -> dict[str, float]:
        """Collect global and footprint-local metrics around (cx, cy)."""
        if err_global is None:
            err_global = float(np.sqrt(np.mean(diff ** 2)))
        err_local = self._local_rmse(diff, cx, cy, self._local_radius)
        rmse_sup = self._surplus_rmse(diff)
        deficit = self._deficit_volume(diff)
        mae = float(np.mean(np.abs(diff)))

        r = self._local_radius
        y0 = max(cy - r, 0)
        y1 = min(cy + r + 1, self._height)
        x0 = max(cx - r, 0)
        x1 = min(cx + r + 1, self._width)
        local_min = float(np.min(diff[y0:y1, x0:x1]))

        loc_surplus, loc_def, loc_grad = self._disk_stats(diff, cx, cy)
        return {
            'rmse_global': float(err_global),
            'rmse_local': float(err_local),
            'rmse_surplus': float(rmse_sup),
            'deficit': float(deficit),
            'mae': mae,
            'local_min': local_min,
            'loc_surplus': float(loc_surplus),
            'loc_def': float(loc_def),
            'loc_grad': float(loc_grad),
            'diff_center': float(diff[cy, cx]),
        }

    # ------------------------------------------------------------------ #
    # Utility: build observation tensor                                  #
    # ------------------------------------------------------------------ #
    def _build_observation(self, diff, h, t):
        """Return the observation tensor using pre-allocated buffers."""
        assert self._env_map is not None

        depth_unit = self._depth_unit

        # 0: signed diff in canonical depth units
        np.divide(diff, depth_unit, out=self._work_diff)
        np.clip(self._work_diff, -5.0, 5.0, out=self._work_diff)

        # 1: env height normalized by running mean, canonical units
        np.subtract(h, self._env_map._mean, out=self._env_norm_buf)
        np.divide(self._env_norm_buf, depth_unit, out=self._env_norm_buf)
        np.clip(self._env_norm_buf, -5.0, 5.0, out=self._env_norm_buf)

        # 2: target height normalized by per-episode mean, canonical units
        np.subtract(t, self._tgt_mean, out=self._tgt_norm_buf)
        np.divide(self._tgt_norm_buf, depth_unit, out=self._tgt_norm_buf)
        np.clip(self._tgt_norm_buf, -5.0, 5.0, out=self._tgt_norm_buf)

        # 3–4: gradient magnitude & Laplacian of canonical diff
        self._update_grad_lap(self._work_diff)

        # Assemble
        self._obs_buf[..., 0] = self._work_diff
        self._obs_buf[..., 1] = self._env_norm_buf
        self._obs_buf[..., 2] = self._tgt_norm_buf
        self._obs_buf[..., 3] = self._grad_buf
        self._obs_buf[..., 4] = self._lap_buf

        # --- Single global scale token (constant plane per episode) ---
        # log-scaled typical diff relative to one max press depth.
        token_scale = float(np.log1p(max(self._s_diff, self._err0) / (depth_unit + self._eps)))
        token_scale = float(np.clip(token_scale, -5.0, 5.0))
        self._obs_buf[..., 5] = token_scale

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

        env_centered = h - self._env_map._mean
        tgt_centered = t - self._tgt_mean
        self._s_env_h = float(max(self._eps, np.percentile(np.abs(env_centered), 95.0)))
        self._s_tgt_h = float(max(self._eps, np.percentile(np.abs(tgt_centered), 95.0)))

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
        self._best_err = self._err0
        self._plateau_counter = 0
        self._success_threshold = max(self._error_threshold,
                                      self._relative_success_frac * self._err0)
        self._plateau_tol = self._plateau_improve_tol * self._err0

        # Build first observation using the new per-episode scales
        obs = self._build_observation(diff, h, t)
        if self.debug:
            self._initial_err0 = self._err0

        return ts.restart(obs)

    # ------------------------------------------------------------------ #
    # One environment step                                               #
    # ------------------------------------------------------------------ #
    def _observe(self, diff):
        h = self._env_map.map
        t = self._target_map.map
        return self._build_observation(diff, h, t)

    def _terminate_episode(self, diff_after, err_g_after, reward):
        self._episode_ended = True
        self._current_err_global = err_g_after
        obs = self._observe(diff_after)
        return ts.termination(obs, reward)

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

        # Match the numba kernels: round-half-up and clamp center so the full tool footprint fits
        r = self._tool_radius
        cy_raw = int(np.floor(y + 0.5))
        cx_raw = int(np.floor(x + 0.5))
        cy = int(np.clip(cy_raw, r, self._patch_height - 1 - r))
        cx = int(np.clip(cx_raw, r, self._patch_width  - 1 - r))
        
        # The press semantics are: tip starts at the local surface (z_abs) and moves down by dz_rel
        z_abs = float(self._env_map.map[cy, cx])
        dz_rel = dz_norm * (self.max_push_mult * self._tool_radius)

        # Global & local metrics before the press
        diff_before, err_g_before = self._diff_and_rmse()
        metrics_before = self._gather_metrics(diff_before, cx, cy, err_global=err_g_before)

        # Prevent unsafe over-cuts: cap depth by local surplus (with small tolerance)
        diff_center = metrics_before['diff_center']
        safe_depth = max(0.0, diff_center + self._alpha_over * self._depth_unit)
        dz_rel = min(dz_rel, safe_depth)

        # Apply press and measure removed volume
        removed, _ = self._env_map.apply_press_abs(x, y, z_abs, dz_rel)
        removed_norm = float(np.clip(removed / (self._max_press_volume + self._eps), 0.0, 1.0))

        # Global & local metrics after the press
        diff_after, err_g_after = self._diff_and_rmse()
        metrics_after = self._gather_metrics(diff_after, cx, cy, err_global=err_g_after)

        reward = self._compute_reward(
            before=metrics_before,
            after=metrics_after,
            removed_norm=removed_norm,
        )

        # Debug scalars (log once so both transition/termination paths share)
        if self.debug:
            self._last_removed_norm = removed_norm
            self._last_removed = removed
            self._last_rel_improve = (err_g_before - err_g_after) / (err_g_before + self._eps)
            self._last_reward = reward
            self._last_err_global = err_g_after
            self._last_err_local = metrics_after['rmse_local']
            self._last_grad = float(np.sqrt(np.mean(self._grad_buf**2)))
            self._last_lap  = float(np.sqrt(np.mean(self._lap_buf**2)))

        # Count this step before applying patience logic
        self._step_count += 1

        # Update best-so-far error and plateau counter
        if (self._best_err - err_g_after) > self._plateau_tol:
            self._best_err = err_g_after
            self._plateau_counter = 0
        elif self._step_count >= self._plateau_min_steps:
            self._plateau_counter += 1

        # ----- early-termination checks -----
        if self._terminate_on_success and err_g_after <= self._success_threshold:
            reward += self._success_bonus
            return self._terminate_episode(diff_after, err_g_after, reward)

        # Use explicit bedrock from HeightMap for breach check
        bedrock = getattr(self._env_map, "bedrock", 0.0)
        if self._fail_on_breach and np.any(self._env_map.map < bedrock - 1e-9):
            reward += self._fail_penalty
            return self._terminate_episode(diff_after, err_g_after, reward)

        # Plateau early-stop: no meaningful improvement for a patience window
        if (self._step_count >= self._plateau_min_steps and
                self._plateau_counter >= self._plateau_patience):
            return self._terminate_episode(diff_after, err_g_after, reward)

        self._current_err_global = err_g_after
        obs = self._observe(diff_after)

        if self._step_count >= self._max_steps:
            self._episode_ended = True
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward, discount=1.0)

    # ---------------------------------------------------- #
    #  Reward (potential-based shaping)                    #
    # ---------------------------------------------------- #
    def _compute_reward(self,
                        *,
                        before: dict[str, float],
                        after: dict[str, float],
                        removed_norm: float) -> float:
        # Relative improvements (normalized by current error, so late-stage presses still get signal)
        denom_rmse = max(before['rmse_global'], self._eps)
        denom_mae = max(before['mae'], self._eps)
        rel_rmse = float(np.clip((before['rmse_global'] - after['rmse_global']) / denom_rmse, -1.0, 1.0))
        rel_mae  = float(np.clip((before['mae']         - after['mae'])         / denom_mae,  -1.0, 1.0))

        # Footprint-local shaping: surplus removal and smoothing
        local_surplus_drop = (before['loc_surplus'] - after['loc_surplus']) / max(self._depth_unit, self._eps)
        local_grad_drop = before['loc_grad'] - after['loc_grad']
        local_deficit_gain = max(0.0, after['loc_def'] - before['loc_def'])
        local_deficit_pen = (
            self._k_local_deficit
            * local_deficit_gain
            / max(self._depth_unit, self._eps)
            * (1.0 + removed_norm)
        )

        # Global deficit and over-cutting penalties
        delta_deficit = after['deficit'] - before['deficit']
        deficit_pen = self._k_deficit * max(0.0, delta_deficit) / max(self._depth_unit, self._eps)

        overcut_before = max(0.0, -(before['diff_center'] + self._alpha_over * self._depth_unit))
        overcut_after = max(0.0, -(after['diff_center'] + self._alpha_over * self._depth_unit))
        overcut_depth = max(0.0, overcut_after - overcut_before)
        overcut_pen = self._k_overcut * (overcut_depth / max(self._depth_unit, self._eps)) * (
            1.0 + self._overcut_vol_scale * removed_norm
        )

        center_deficit_pen = self._k_center_deficit * max(0.0, -before['diff_center']) / max(self._depth_unit, self._eps)

        improve = (
            self._lambda_rmse * rel_rmse
            + self._lambda_mae * rel_mae
            + self._lambda_local_surplus * local_surplus_drop
            + self._lambda_local_grad * local_grad_drop
        )

        reward = (
            self._reward_scale * improve
            - deficit_pen
            - local_deficit_pen
            - overcut_pen
            - self._vol_reg * removed_norm
            - center_deficit_pen
        )

        if self.debug:
            self._last_reward_terms = {
                'rel_rmse': rel_rmse,
                'rel_mae': rel_mae,
                'local_surplus_drop': float(local_surplus_drop),
                'local_grad_drop': float(local_grad_drop),
                'improve': float(improve),
                'deficit_pen': float(deficit_pen),
                'local_deficit_pen': float(local_deficit_pen),
                'overcut_pen': float(overcut_pen),
                'center_deficit_pen': float(center_deficit_pen),
                'removed_norm': float(removed_norm),
                'delta_deficit': float(delta_deficit),
                'overcut_depth': float(overcut_depth),
                'overcut_before': float(overcut_before),
                'overcut_after': float(overcut_after),
            }
            self._last_reward = float(reward)
            self._last_err_global = float(after['rmse_global'])
            self._last_err_local = float(after['rmse_local'])
            self._last_rel_improve = float(improve)
        return float(reward)
