import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from heightmap import HeightMap

class SandShapingEnv(py_environment.PyEnvironment):
    def __init__(self,
                 width=128,
                 height=128,
                 patch_width=128,
                 patch_height=128,
                 scale_range=(1, 2),
                 target_scale_range=(2, 4),
                 amplitude_range=(10.0, 40.0),
                 tool_radius=10,
                 max_steps=200,
                 alpha=0.5,
                 error_threshold=0.05,
                 success_bonus=1.0,
                 fail_penalty=-1.0,
                 terminate_on_success=True,
                 fail_on_breach=True,
                 progress_only=False,
                 debug=False,
                 seed=None):
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
        self._amp_max = self._amplitude_range[1]  # cache to avoid repeated index lookups
        # Precompute reciprocals for observation scaling
        self._inv_scale_d = 2.0 / self._amp_max
        self._inv_scale_h = 1.0 / self._amp_max
        self._tool_radius = tool_radius
        self._max_steps = max_steps
        self._alpha = alpha

        # Success / failure logic
        self._error_threshold      = error_threshold
        self._success_bonus        = success_bonus
        self._fail_penalty         = fail_penalty
        self._terminate_on_success = terminate_on_success
        self._fail_on_breach       = fail_on_breach

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

        self._episode_ended = False

        self._env_map = None
        self._target_map = None

        self.reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


    # ---------------------------------------------------- #
    #  Reward computation – easy to swap for new schemes   #
    # ---------------------------------------------------- #
    def _compute_reward(self,
                        err_before, err_after,
                        loc_err_before, loc_err_after):
        """Return scalar reward for a press action."""
        delta_glob = err_before - err_after
        delta_loc  = loc_err_before - loc_err_after
        return self._alpha * delta_glob + (1.0 - self._alpha) * delta_loc
    
    # ------------------------------------------------------------------ #
    # Utility: build 3‑channel observation and (optionally) visualise it #
    # ------------------------------------------------------------------ #
    def _build_observation(self, diff, h, t):
        """Return the 3‑channel observation tensor.

        Parameters
        ----------
        diff : np.ndarray
            Difference map (env - target), same shape as `h`.
        h : np.ndarray
            Current environment height map.
        t : np.ndarray
            Target height map.

        Returns
        -------
        obs : np.ndarray, float32, shape (H, W, 3), range [-1,1]
        """
        diff_signed = np.clip(diff * self._inv_scale_d, -1.0, 1.0)
        env_norm    = np.clip((h - h.mean()) * self._inv_scale_h, -1.0, 1.0)
        tgt_norm    = np.clip((t - t.mean()) * self._inv_scale_h, -1.0, 1.0)

        obs = np.stack([diff_signed, env_norm, tgt_norm], axis=-1).astype(np.float32)

        return obs

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

        self._step_count = 0
        self._episode_ended = False

        # Compute initial difference and normalize to [-1,1]
        diff = self._env_map.difference(self._target_map)
        # Build 3-channel observation:
        h = self._env_map.map
        t = self._target_map.map
        obs = self._build_observation(diff, h, t)
        return ts.restart(obs)

    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        # Parse action
        x_norm, y_norm, dz_norm = action

        x = self._tool_radius + x_norm * (self._width  - 2 * self._tool_radius)
        y = self._tool_radius + y_norm * (self._height - 2 * self._tool_radius)

        # Derive absolute tip height from current surface, then push down by dz_norm
        cy = int(np.clip(round(y), self._tool_radius, self._height - 1 - self._tool_radius))
        cx = int(np.clip(round(x), self._tool_radius, self._width  - 1 - self._tool_radius))
        h_center = float(self._env_map.map[cy, cx])
        z_abs = h_center
        dz_rel = dz_norm * (0.66 * self._tool_radius)

        # Local RMSE before the press
        diff_before = self._env_map.difference(self._target_map)
        err_before = np.sqrt(np.mean(diff_before**2))
        # Local RMSE before the press
        r = self._tool_radius
        y0, y1 = cy - r, cy + r + 1
        x0, x1 = cx - r, cx + r + 1
        sub_before = diff_before[y0:y1, x0:x1]
        mask = self._env_map._press_mask
        loc_err_before = np.sqrt(np.mean(sub_before[mask]**2))

        # Apply press and measure removed volume
        removed, touched = self._env_map.apply_press_abs(x, y, z_abs, dz_rel)

        # Local RMSE after the press
        diff_after = self._env_map.difference(self._target_map)
        sub_after = diff_after[y0:y1, x0:x1]
        loc_err_after = np.sqrt(np.mean(sub_after[mask]**2))
        err_after = np.sqrt(np.mean(diff_after**2))
        reward = self._compute_reward(err_before, err_after,
                                      loc_err_before, loc_err_after)

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