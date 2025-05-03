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
                 amplitude_range=(1.0, 20.0),
                 tool_radius=20,
                 max_steps=100):
        self._width = width
        self._height = height
        self._patch_width = patch_width
        self._patch_height = patch_height
        self._scale_range = scale_range
        self._target_scale_range = target_scale_range
        self._amplitude_range = amplitude_range
        self._tool_radius = tool_radius
        self._max_steps = max_steps

        # Action spec: [x, y, z_start, dz]
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(4,),
            dtype=np.float32,
            minimum=np.array([0, 0, 0, 0], dtype=np.float32),
            maximum=np.array([self._width - 1,
                               self._height - 1,
                               self._amplitude_range[1],
                               self._amplitude_range[1]], dtype=np.float32),
            name='action'
        )

        # Observation spec: registered difference patch
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._patch_height, self._patch_width),
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
        scale_x = np.random.randint(self._scale_range[0], self._scale_range[1] + 1)
        scale_y = np.random.randint(self._scale_range[0], self._scale_range[1] + 1)
        amplitude = np.random.uniform(self._amplitude_range[0], self._amplitude_range[1])
        self._env_map = HeightMap(self._width,
                                  self._height,
                                  scale=(scale_x, scale_y),
                                  amplitude=amplitude,
                                  tool_radius=self._tool_radius)

        # Sample new target patch
        tgt_scale_x = np.random.randint(self._target_scale_range[0], self._target_scale_range[1] + 1)
        tgt_scale_y = np.random.randint(self._target_scale_range[0], self._target_scale_range[1] + 1)
        tgt_amplitude = np.random.uniform(self._amplitude_range[0], self._amplitude_range[1])
        self._target_map = HeightMap(self._patch_width,
                                     self._patch_height,
                                     scale=(tgt_scale_x, tgt_scale_y),
                                     amplitude=tgt_amplitude,
                                     tool_radius=self._tool_radius)

        self._step_count = 0
        self._episode_ended = False

        # Compute initial difference and normalize to [-1,1]
        diff, _ = self._env_map.registered_difference(self._target_map, zero_center=True)
        obs = (diff / self._amplitude_range[1]).astype(np.float32)
        return ts.restart(obs)

    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        # Parse action
        x, y, z_start, dz = action
        # Convert normalized heights back to map units
        z_max = self._env_map.amplitude
        z0 = z_start * z_max
        dz0 = dz * z_max

        self._env_map.apply_press(x, y, z0, dz0)

        # Compute reward
        reward, _ = self._env_map.compute_reward(self._target_map,
                                                 zero_center=True,
                                                 method='l2')
        self._step_count += 1

        # Compute new difference and normalize to [-1,1]
        diff, _ = self._env_map.registered_difference(self._target_map, zero_center=True)
        obs = (diff / self._amplitude_range[1]).astype(np.float32)
        if self._step_count >= self._max_steps:
            self._episode_ended = True
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward, discount=1.0)
