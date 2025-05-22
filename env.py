import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from heightmap import HeightMap

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
        tool_radius: int = 10,
        max_steps: int = 200,
        # ── TERMINATION & OUTCOME BONUSES ─────────────────────────
        error_threshold: float = 0.05,
        success_bonus: float = 1.0,
        fail_penalty: float = -1.0,
        terminate_on_success: bool = True,
        fail_on_breach: bool = True,
        # ── REWARD SHAPING / SCALING KNOBS ────────────────────────
        volume_penalty_coeff: float = 3e-4,
        no_touch_penalty: float = 0.05,
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
        self._amp_max = self._amplitude_range[1]  # cache to avoid repeated index lookups
        # Precompute reciprocals for observation scaling
        self._inv_scale_d = 2.0 / self._amp_max
        self._inv_scale_h = 1.0 / self._amp_max

        # ── TOOL / ACTION & EPISODE HORIZON ───────────────────────
        self._tool_radius = tool_radius
        self._max_steps = max_steps

        # ── TERMINATION & OUTCOME BONUSES ─────────────────────────
        self._error_threshold      = error_threshold
        self._success_bonus        = success_bonus
        self._fail_penalty         = fail_penalty
        self._terminate_on_success = terminate_on_success
        self._fail_on_breach       = fail_on_breach

        # ── REWARD SHAPING / SCALING KNOBS ────────────────────────
        self._volume_penalty_coeff = volume_penalty_coeff
        self._no_touch_penalty     = no_touch_penalty
        self._eps                  = 1e-6
        self._progress_bonus = 0.05  # extra reward when a new best error is reached
        self._best_err = np.inf      # will be initialised per‑episode in _reset

        # Efficiency–aware volume shaping
        self._efficiency_coeff     = 0.10   # reward scale for efficient carving
        self._waste_penalty_coeff  = 0.05   # extra penalty for wasted volume
        # Maximum volume removable by a single press (for normalisation)
        self._tool_area = np.pi * (self._tool_radius ** 2)
        self._max_press_volume = self._tool_area * (0.66 * self._tool_radius)

        # ── INTERNAL WORK BUFFERS (pre‑allocated, no per‑step realloc) ──
        self._work_diff      = np.empty((self._patch_height, self._patch_width), dtype=np.float32)
        self._env_norm_buf   = np.empty_like(self._work_diff)
        self._tgt_norm_buf   = np.empty_like(self._work_diff)
        self._obs_buf        = np.empty((self._patch_height, self._patch_width, 3), dtype=np.float32)

        # ── SPECS ────────────────────────────────────────────────
        # Action spec: [x, y, dz_rel]  all normalised to [0,1]
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,),
            dtype=np.float32,
            minimum=np.array([0, 0, 0], dtype=np.float32),
            maximum=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            name='action'
        )

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._patch_height, self._patch_width, 3),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='observation'
        )

        # ── STATE VARIABLES ──────────────────────────────────────
        self._episode_ended = False
        self._env_map = None
        self._target_map = None
        self._tgt_mean = 0.0   # cached mean of target map for fast normalisation

        self.reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


    # ---------------------------------------------------- #
    #  Reward computation – easy to swap for new schemes   #
    # ---------------------------------------------------- #
    def _compute_reward(self,
                        err_before: float,
                        err_after: float,
                        removed: float,
                        touched: bool) -> float:
        """
        Compute shaped reward for a single press.

        Components
        ----------
        1. Relative improvement           : (err_before - err_after) / err_before
        2. Progress bonus                 : awarded only if improvement > 1 % of err_before
        3. Linear volume penalty          : discourages digging when no gain is made
        4. Efficiency bonus               : improvement per unit normalised volume
        5. Waste penalty                  : extra cost when removed‑volume exceeds improvement
        6. No‑touch penalty               : fixed penalty when the press missed the surface
        The final reward is clipped to [-1, 1] for stability.
        """
        # --- 1) Relative improvement (self‑normalised) ----------------------
        rel_improve = (err_before - err_after) / (err_before + self._eps)
        reward = rel_improve

        # --- 2) Progress bonus (only if >1 % improvement) -------------------
        if err_before - err_after > 0.01 * err_before:
            reward += self._progress_bonus

        # --- 3) Volume‑based shaping ---------------------------------------
        # Normalised removed volume in [0,1]
        vol_norm = removed / (self._max_press_volume + self._eps)
        # Base linear penalty
        reward -= self._volume_penalty_coeff * vol_norm

        # --- 4) Efficiency term: improvement per unit volume ---------------
        efficiency = rel_improve / (vol_norm + self._eps)
        reward += self._efficiency_coeff * efficiency

        # --- 5) Waste penalty ----------------------------------------------
        # If more volume removed than improvement achieved, penalise the excess
        waste = max(vol_norm - rel_improve, 0.0)
        reward -= self._waste_penalty_coeff * waste

        # --- 6) Miss penalty ------------------------------------------------
        if not touched:
            reward -= self._no_touch_penalty

        # --- Final safety‑clip ---------------------------------------------
        reward = float(np.clip(reward, -1.0, 1.0))
        return reward
    
    # ------------------------------------------------------------------ #
    # Utility: build 3‑channel observation and (optionally) visualise it #
    # ------------------------------------------------------------------ #
    def _build_observation(self, diff, h, t):
        """Return the 3‑channel observation tensor using pre‑allocated buffers."""
        # Channel 0  – signed difference, scaled
        np.multiply(diff, self._inv_scale_d, out=self._work_diff)
        np.clip(self._work_diff, -1.0, 1.0, out=self._work_diff)

        # Channel 1  – normalised current height (use running mean from HeightMap)
        np.subtract(h, self._env_map._mean, out=self._env_norm_buf)
        self._env_norm_buf *= self._inv_scale_h
        np.clip(self._env_norm_buf, -1.0, 1.0, out=self._env_norm_buf)

        # Channel 2  – normalised target height (cached per episode)
        np.subtract(t, self._tgt_mean, out=self._tgt_norm_buf)
        self._tgt_norm_buf *= self._inv_scale_h
        np.clip(self._tgt_norm_buf, -1.0, 1.0, out=self._tgt_norm_buf)

        # Assemble into the shared observation buffer without extra allocations
        self._obs_buf[..., 0] = self._work_diff
        self._obs_buf[..., 1] = self._env_norm_buf
        self._obs_buf[..., 2] = self._tgt_norm_buf
        return self._obs_buf

    # ------------------------------------------------------------------ #
    # Episode initialisation: sample new terrain & target, reset metrics #
    # ------------------------------------------------------------------ #
    def _reset(self):
        # Sample new substrate
        scale_x = self._rng.uniform(self._scale_range[0], self._scale_range[1])
        scale_y = self._rng.uniform(self._scale_range[0], self._scale_range[1])
        amplitude = self._rng.uniform(self._amplitude_range[0], self._amplitude_range[1])
        env_seed = None if self._seed is None else int(self._rng.integers(0, 2**32))
        self._env_map = HeightMap(self._width,
                                  self._height,
                                  scale=(scale_x, scale_y),
                                  amplitude=amplitude,
                                  tool_radius=self._tool_radius,
                                  seed=env_seed,
                                  bedrock_offset=30)

        # Sample new target patch
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
        # Cache target mean for the episode
        self._tgt_mean = self._target_map._mean

        self._step_count = 0
        self._episode_ended = False

        # Compute initial difference and normalize to [-1,1]
        diff = self._env_map.difference(self._target_map, out=self._work_diff)
        # Build 3-channel observation:
        h = self._env_map.map
        t = self._target_map.map
        obs = self._build_observation(diff, h, t)

        # Cache initial global RMSE for per‑episode normalisation
        self._err0 = float(np.sqrt(np.mean(diff**2))) + self._eps
        self._best_err = self._err0

        return ts.restart(obs)


    # ------------------------------------------------------------------ #
    # One environment step: execute press, update reward & termination   #
    # ------------------------------------------------------------------ #
    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        # Parse action
        x_norm, y_norm, dz_norm = action

        # Map normalised action to world coordinates / absolute tool tip
        x = self._tool_radius + x_norm * (self._width  - 2 * self._tool_radius)
        y = self._tool_radius + y_norm * (self._height - 2 * self._tool_radius)

        # Derive absolute tip height from current surface, then push down by dz_norm
        cy = int(np.clip(round(y), self._tool_radius, self._height - 1 - self._tool_radius))
        cx = int(np.clip(round(x), self._tool_radius, self._width  - 1 - self._tool_radius))
        h_center = float(self._env_map.map[cy, cx])
        z_abs = h_center
        dz_rel = dz_norm * (0.66 * self._tool_radius)

        # Local RMSE before the press
        diff_before = self._env_map.difference(self._target_map, out=self._work_diff)
        err_before = np.sqrt(np.mean(diff_before**2))

        # Apply press and measure removed volume
        removed, touched = self._env_map.apply_press_abs(x, y, z_abs, dz_rel)

        # Local RMSE after the press
        diff_after = self._env_map.difference(self._target_map, out=self._work_diff)
        err_after = np.sqrt(np.mean(diff_after**2))
        reward = self._compute_reward(err_before, err_after, removed, touched)

        # ----- early‑termination checks -----
        # 1) success if global RMSE is sufficiently low
        if self._terminate_on_success and err_after <= self._error_threshold:
            self._episode_ended = True
            reward += self._success_bonus
            h = self._env_map.map
            t = self._target_map.map
            obs = self._build_observation(diff_after, h, t)
            return ts.termination(obs, reward)

        # 2) failure if any voxel would carve below bedrock
        if self._fail_on_breach and np.any((diff_after + self._env_map.map) < 0.0):
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
        else:
            return ts.transition(obs, reward, discount=1.0)