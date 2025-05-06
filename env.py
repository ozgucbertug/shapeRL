import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from heightmap import HeightMap

class SandShapingEnv(py_environment.PyEnvironment):
    def __init__(self,
                 width=400,
                 height=400,
                 patch_width=400,
                 patch_height=400,
                 scale_range=(1, 2),
                 target_scale_range=(2, 4),
                 amplitude_range=(10.0, 40.0),
                 tool_radius=25,
                 max_steps=100):
        self._width = width
        self._height = height
        self._patch_width = patch_width
        self._patch_height = patch_height
        self._scale_range = scale_range
        self._target_scale_range = target_scale_range
        self._amplitude_range = amplitude_range
        self._inv_amplitude_max = 1.0 / self._amplitude_range[1]
        self._tool_radius = tool_radius
        self._max_steps = max_steps

        # Action spec: [x, y, dz]
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,),
            dtype=np.float32,
            minimum=np.array([0, 0, 0], dtype=np.float32),
            maximum=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            name='action'
        )

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._patch_height, self._patch_width, 1),
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

    def _reset(self):
        # Sample new substrate
        scale_x = np.random.uniform(self._scale_range[0], self._scale_range[1])
        scale_y = np.random.uniform(self._scale_range[0], self._scale_range[1])
        amplitude = np.random.uniform(self._amplitude_range[0], self._amplitude_range[1])
        self._env_map = HeightMap(self._width,
                                  self._height,
                                  scale=(scale_x, scale_y),
                                  amplitude=amplitude,
                                  tool_radius=self._tool_radius)

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
        # Compute initial total error for reward normalization
        self._initial_error = np.sqrt(np.sum(np.square(diff)))
        if self._initial_error == 0:
            self._initial_error = 1.0
        obs = (diff * self._inv_amplitude_max).astype(np.float32)
        obs = obs[..., np.newaxis]
        return ts.restart(obs)

    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        # Parse action
        x_norm, y_norm, dz = action

        x = self._tool_radius + x_norm * (self._width - 2*self._tool_radius)
        y = self._tool_radius + y_norm * (self._height - 2*self._tool_radius)
        dz0 = dz * (self._tool_radius * 0.66)

        # Compute pre-press global error
        diff_before = self._env_map.difference(self._target_map)
        err_before = np.sqrt(np.sum(np.square(diff_before)))

        # Apply press
        self._env_map.apply_press(x, y, dz0)

        # Compute post-press global error and reward
        diff_after = self._env_map.difference(self._target_map)
        err_after = np.sqrt(np.sum(np.square(diff_after)))
        reward = err_before - err_after

        print(reward)

        self._step_count += 1

        # Build observation from diff_after using precomputed inverse amplitude
        obs = (diff_after * self._inv_amplitude_max).astype(np.float32)
        obs = obs[..., np.newaxis]
        if self._step_count >= self._max_steps:
            self._episode_ended = True
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward, discount=1.0)