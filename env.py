import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from heightmap import HeightMap

import matplotlib.pyplot as plt

class SandShapingEnv(py_environment.PyEnvironment):
    def __init__(self,
                 width=128,
                 height=128,
                 patch_width=128,
                 patch_height=128,
                 scale_range=(1, 2),
                 target_scale_range=(2, 4),
                 amplitude_range=(10.0, 40.0),
                 tool_radius=20,
                 max_steps=200,
                 alpha=0.5,
                 progress_only=False,
                 debug=False):
        self.debug = debug
        self._width = width
        self._height = height
        self._patch_width = patch_width
        self._patch_height = patch_height
        self._scale_range = scale_range
        self._target_scale_range = target_scale_range
        self._amplitude_range = amplitude_range
        self._amp_max = self._amplitude_range[1]  # cache to avoid repeated index lookups
        # Penalty coefficient per unit volume removed
        self._vol_penalty = 0.1
        # Fixed penalty for presses that change nothing
        self._zero_penalty = 0.1
        # Penalty for a press that contacts nothing
        self._no_touch_penalty = 0.1
        self._tool_radius = tool_radius
        self._max_steps = max_steps
        self._alpha = alpha
        self._progress_only = progress_only

        # Action spec: [x, y, z_abs, dz_rel]  all normalised to [0,1]
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(4,),
            dtype=np.float32,
            minimum=np.array([0, 0, 0, 0], dtype=np.float32),
            maximum=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
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
        s = 0.5 * self._amp_max          # symmetric squash for difference
        scale_h = self._amp_max          # linear clip for height channels

        obs = np.stack([
            np.tanh(diff / s),
            np.clip((h - h.mean()) / scale_h, -1.0, 1.0),
            np.clip((t - t.mean()) / scale_h, -1.0, 1.0)
        ], axis=-1).astype(np.float32)

        # One‑time visualisation controlled by self.debug
        if self.debug and not getattr(self, "_first_obs_shown", False):
            channel_names = ['difference', 'current height', 'target height']
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i, name in enumerate(channel_names):
                ax = axes[i]
                im = ax.imshow(obs[..., i], cmap='turbo',
                               vmin=obs[..., i].min(), vmax=obs[..., i].max())
                ax.set_title(f'{name} channel (first obs)')
                fig.colorbar(im, ax=ax, label='value')
            plt.show(block=True)
            self._first_obs_shown = True
        return obs

    def _reset(self):
        # Sample new substrate
        scale_x = np.random.uniform(self._scale_range[0], self._scale_range[1])
        scale_y = np.random.uniform(self._scale_range[0], self._scale_range[1])
        amplitude = np.random.uniform(self._amplitude_range[0], self._amplitude_range[1])
        self._env_map = HeightMap(self._width,
                                  self._height,
                                  scale=(scale_x, scale_y),
                                  amplitude=amplitude,
                                  tool_radius=self._tool_radius,
                                  bedrock_offset=30)

        # Sample new target patch
        tgt_scale_x = np.random.uniform(self._target_scale_range[0], self._target_scale_range[1])
        tgt_scale_y = np.random.uniform(self._target_scale_range[0], self._target_scale_range[1])
        tgt_amplitude = np.random.uniform(self._amplitude_range[0], self._amplitude_range[1])
        self._target_map = HeightMap(self._patch_width,
                                     self._patch_height,
                                     scale=(tgt_scale_x, tgt_scale_y),
                                     amplitude=tgt_amplitude,
                                     tool_radius=self._tool_radius)

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
        x_norm, y_norm, z_norm, dz_norm = action

        x = self._tool_radius + x_norm * (self._width  - 2 * self._tool_radius)
        y = self._tool_radius + y_norm * (self._height - 2 * self._tool_radius)

        # Absolute tool‑tip height in world Z
        z_abs = z_norm * (self._env_map.amplitude + self._env_map.bedrock_offset)
        dz_rel = dz_norm * (0.66 * self._env_map.amplitude)

        # Local RMSE before the press
        diff_before = self._env_map.difference(self._target_map)
        cy = int(np.clip(round(y), self._tool_radius, self._height - 1 - self._tool_radius))
        cx = int(np.clip(round(x), self._tool_radius, self._width  - 1 - self._tool_radius))
        coords = np.indices(diff_before.shape)
        mask_local = (coords[0] - cy)**2 + (coords[1] - cx)**2 <= self._tool_radius**2
        loc_err_before = np.sqrt(np.mean(diff_before[mask_local]**2))

        # Apply press and measure removed volume
        removed, touched = self._env_map.apply_press_abs(x, y, z_abs, dz_rel)

        # Local RMSE after the press
        diff_after = self._env_map.difference(self._target_map)
        loc_err_after = np.sqrt(np.mean(diff_after[mask_local]**2))

        # Reward = reduction in local RMSE
        reward = loc_err_before - loc_err_after

        # Log to TensorBoard
        tf.summary.scalar('env/removed_volume', removed, step=self._step_count)
        tf.summary.scalar('env/touched_flag', float(touched), step=self._step_count)
        tf.summary.scalar('env/delta_local', reward, step=self._step_count)

        self._step_count += 1

        h = self._env_map.map
        t = self._target_map.map
        obs = self._build_observation(diff_after, h, t)
        if self._step_count >= self._max_steps:
            self._episode_ended = True
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward, discount=1.0)