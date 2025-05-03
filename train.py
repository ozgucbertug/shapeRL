import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import os
from matplotlib import patches
from scipy.signal import correlate2d
from scipy.ndimage import rotate

def fade(t):
    """
    Fade function for Perlin noise.
    """
    return 6*t**5 - 15*t**4 + 10*t**3

def generate_perlin_noise_2d(shape, res, amplitude=1.0):
    """
    Generate a 2D Perlin noise heightmap.
    :param shape: Tuple (height, width) of the output array.
    :param res: Number of noise periods along each axis as (res_x, res_y).
    :param amplitude: Amplitude to scale the noise.
    :return: 2D numpy array of shape `shape`, values in [0, amplitude].
    """
    height, width = shape
    # Create coordinate grid in noise space
    xs = np.linspace(0, res[0], width, endpoint=False)
    ys = np.linspace(0, res[1], height, endpoint=False)
    xv, yv = np.meshgrid(xs, ys)

    # Integer grid cell coordinates
    xi = np.floor(xv).astype(int)
    yi = np.floor(yv).astype(int)
    # Local coordinates within cell
    xf = xv - xi
    yf = yv - yi

    # Compute fade curves
    u = fade(xf)
    v = fade(yf)

    # Random gradient vectors at each grid corner
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def dot_grad(ix, iy, x, y):
        g = gradients[ix % (res[0] + 1), iy % (res[1] + 1)]
        return g[..., 0] * x + g[..., 1] * y

    # Dot products at the four corners
    n00 = dot_grad(xi,   yi,   xf,   yf)
    n10 = dot_grad(xi+1, yi,   xf-1, yf)
    n01 = dot_grad(xi,   yi+1, xf,   yf-1)
    n11 = dot_grad(xi+1, yi+1, xf-1, yf-1)

    # Linear interpolation
    x1 = n00 * (1 - u) + n10 * u
    x2 = n01 * (1 - u) + n11 * u
    noise = x1 * (1 - v) + x2 * v

    # Normalize result to [0, 1] and multiply by amplitude
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise * amplitude

class HeightMap:
    """
    Heightmap representation using a 2D array.
    """
    def __init__(self, width, height, scale=(4, 4), amplitude=1.0, tool_radius=5, seed=None):
        """
        Initialize a heightmap with Perlin noise.
        :param width: Number of columns.
        :param height: Number of rows.
        :param scale: Tuple (res_x, res_y) controlling noise frequency.
        :param amplitude: Amplitude scale for noise.
        :param tool_radius: Radius of the spherical end-effector tool.
        :param seed: Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        self.width = width
        self.height = height
        self.tool_radius = tool_radius
        self.scale = scale
        self.amplitude = amplitude
        self.map = generate_perlin_noise_2d((height, width), scale, amplitude)

    def apply_press(self, x, y, z_start, dz, theta=0, phi=0, d=0):
        """
        Simulate a spherical press along world Z.
        :param x: X-coordinate (column) of press center.
        :param y: Y-coordinate (row) of press center.
        :param z_start: Starting height of sphere center (normalized 0-1).
        :param dz: Depth to press (positive value moves sphere down).
        :param theta: Unused placeholder for rotation around X.
        :param phi: Unused placeholder for rotation around Y.
        :param d: Unused placeholder for sliding distance.
        """
        z_end = z_start - dz
        r = self.tool_radius
        # bounding box
        y_min = max(0, int(np.floor(y - r)))
        y_max = min(self.height, int(np.ceil(y + r)) + 1)
        x_min = max(0, int(np.floor(x - r)))
        x_max = min(self.width, int(np.ceil(x + r)) + 1)
        # coordinate grids
        yy = np.arange(y_min, y_max)[:, None]
        xx = np.arange(x_min, x_max)[None, :]
        dx2 = (xx - x)**2
        dy2 = (yy - y)**2
        # compute intersection heights only within the tool radius
        dist2 = dx2 + dy2
        inside = dist2 <= r**2
        z_int = np.zeros_like(dist2, dtype=float)
        z_int[inside] = z_end - np.sqrt(r**2 - dist2[inside])
        # apply to submap
        sub = self.map[y_min:y_max, x_min:x_max]
        sub[inside] = np.minimum(sub[inside], z_int[inside])
        self.map[y_min:y_max, x_min:x_max] = sub

    def to_grayscale_image(self):
        """
        Convert the heightmap to a uint8 grayscale image (0-255).
        """
        # Normalize to [0,1]
        h = self.map
        h_min, h_max = h.min(), h.max()
        norm = (h - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(h)
        # Scale to [0,255]
        return (norm * 255).astype(np.uint8)

    def to_rgb_image(self, cmap='terrain'):
        """
        Convert the heightmap to an RGB image using a matplotlib colormap.
        :param cmap: Colormap name.
        :return: (H, W, 3) uint8 array.
        """
        norm = (self.map - self.map.min()) / (self.map.max() - self.map.min()) if self.map.max() > self.map.min() else np.zeros_like(self.map)
        cm = plt.get_cmap(cmap)
        rgb = cm(norm)[..., :3]  # Drop alpha channel
        return (rgb * 255).astype(np.uint8)

    def difference(self, other, zero_center=False):
        """
        Compute the per-pixel height difference between this heightmap and another.
        :param other: Another HeightMap instance or a 2D array of the same shape.
        :param zero_center: If True, subtract the mean difference so that the result is centered around zero.
        :return: A 2D numpy array of the same shape containing the difference.
        """
        # Extract the other height data
        if isinstance(other, HeightMap):
            h2 = other.map
        else:
            h2 = np.asarray(other)
        # Compute raw difference
        diff = self.map - h2
        # Optionally center around zero
        if zero_center:
            diff = diff - np.mean(diff)
        return diff

    def reset(self, seed=None):
        """
        Reset the heightmap to a new random Perlin noise state.
        :param seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        self.map = generate_perlin_noise_2d(
            (self.height, self.width), self.scale, self.amplitude)

    def get_state(self, as_rgb=False, cmap='terrain'):
        """
        Return the current heightmap as an image suitable for RL observation.
        :param as_rgb: If True, returns an RGB image; otherwise, grayscale.
        :param cmap: Colormap name for RGB output.
        :return: numpy array of shape (H, W) for grayscale or (H, W, 3) for RGB.
        """
        if as_rgb:
            return self.to_rgb_image(cmap)
        else:
            return self.to_grayscale_image()

    def compute_reward(self, target, zero_center=True, method='l2'):
        """
        Compute a scalar reward comparing this heightmap to a target patch.
        :param target: Another HeightMap or 2D array to match.
        :param zero_center: Whether to zero-center the difference.
        :param method: 'l2' (negative Euclidean norm), 'l1' (negative sum of abs).
        :return: (reward, (x_offset, y_offset)) tuple, where reward is higher (less negative)
                 when the patch better matches the target.
        """
        diff = self.difference(target, zero_center=zero_center)
        if method == 'l2':
            reward = -np.linalg.norm(diff)
        elif method == 'l1':
            reward = -np.sum(np.abs(diff))
        else:
            raise ValueError(f"Unsupported method: {method}")
        return reward

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class SandShapingEnv(py_environment.PyEnvironment):
    def __init__(self,
                 width=400,
                 height=400,
                 patch_width=400,
                 patch_height=400,
                 scale_range=(1, 2),
                 target_scale_range=(2, 4),
                 amplitude_range=(1.0, 20.0),
                 tool_radius=50,
                 max_steps=200):
        self._width = width
        self._height = height
        self._patch_width = patch_width
        self._patch_height = patch_height
        # Ensure the target patch fits within the environment
        if self._patch_width > self._width or self._patch_height > self._height:
            raise ValueError(
                f"Target patch size ({self._patch_width}x{self._patch_height}) exceeds environment size ({self._width}x{self._height})."
            )
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

        # Observation spec: registered difference patch with channel dimension
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
        diff = self._env_map.difference(self._target_map, zero_center=True)
        obs = (diff / self._amplitude_range[1]).astype(np.float32)
        obs = obs[..., np.newaxis]
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

        # Track surface before pressing to detect no-op
        prev_map = self._env_map.map.copy()
        self._env_map.apply_press(x, y, z0, dz0)

        # Compute L2 norm of the difference and normalize to [0,1]
        # compute_reward returns a scalar reward = -l2_norm
        reward = -self._env_map.compute_reward(
            self._target_map, zero_center=True, method='l2')
        self._step_count += 1

        # Compute new difference and normalize to [-1,1]
        diff = self._env_map.difference(self._target_map, zero_center=True)
        obs = (diff / self._amplitude_range[1]).astype(np.float32)
        obs = obs[..., np.newaxis]
        if self._step_count >= self._max_steps:
            self._episode_ended = True
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward, discount=1.0)
        

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.utils.common import function, Checkpointer
from tensorflow.keras.layers import Conv2D, Resizing

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action = policy.action(time_step).action
            time_step = environment.step(action)
            episode_return += time_step.reward
        total_return += episode_return
    return total_return / num_episodes

def train():
    # Hyperparameters
    num_iterations = 200000
    collect_steps_per_iteration = 1
    replay_buffer_capacity = 100000
    batch_size = 16
    learning_rate = 3e-4
    gamma = 0.99
    eval_interval = 10000
    num_eval_episodes = 5
    # Visualization settings
    vis_interval = 50

    # Create environments
    train_py_env = SandShapingEnv()
    eval_py_env = SandShapingEnv()
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Define networks
    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        conv_layer_params=((16, 3, 2), (32, 3, 2)),
        fc_layer_params=(256, 128)
    )
    critic_net = CriticNetwork(
        input_tensor_spec=(observation_spec, action_spec),
        observation_conv_layer_params=((16, 3, 2), (32, 3, 2)),
        action_fc_layer_params=None,
        joint_fc_layer_params=(256, 128),
        name='critic_network'
    )

    # Create SAC agent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf_agent = sac_agent.SacAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate),
        critic_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate),
        alpha_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate),
        target_update_tau=0.005,
        target_update_period=1,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=1.0,
        train_step_counter=global_step
    )
    tf_agent.initialize()

    # Replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity
    )
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2,
        single_deterministic_pass=False
    ).prefetch(3)
    iterator = iter(dataset)

    # Data collection driver
    collect_driver = DynamicStepDriver(
        train_env,
        tf_agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration
    )

    # Metrics and evaluation
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    # (Optional) Checkpointer
    checkpoint_dir = 'checkpoints'
    train_checkpointer = Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=tf_agent,
        policy=tf_agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    train_checkpointer.initialize_or_restore()

    # Warm up replay buffer before training
    initial_collect_steps = batch_size + 1
    for _ in range(initial_collect_steps):
        collect_driver.run()

    # Set up non-blocking Matplotlib visualization
    fig_vis, axes_vis = plt.subplots(1, 3, figsize=(15, 5))

    # Training loop
    tf_agent.train = function(tf_agent.train)
    collect_driver.run = function(collect_driver.run)

    for _ in range(num_iterations):
        # Collect data
        collect_driver.run()
        # Sample a batch and train
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience).loss
        step = tf_agent.train_step_counter.numpy()
        if step % 1 == 0:
            print(f'step={step}: loss={train_loss.numpy():.4f}')

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print(f'step={step}: avg_return={avg_return:.2f}')


        # Non-blocking Matplotlib visualization every vis_interval steps
        if step % vis_interval == 0:
            env_img = train_py_env._env_map.map
            target_img = train_py_env._target_map.map
            diff_img = env_img - target_img
            axes_vis[0].clear()
            axes_vis[0].imshow(env_img, cmap='viridis')
            axes_vis[0].set_title(f'Env @ step {step}')
            axes_vis[1].clear()
            axes_vis[1].imshow(target_img, cmap='viridis')
            axes_vis[1].set_title('Target')
            axes_vis[2].clear()
            axes_vis[2].imshow(diff_img, cmap='viridis')
            axes_vis[2].set_title('Difference')
            fig_vis.canvas.draw()
            plt.pause(0.001)

    # Save final policy
    policy_dir = 'policy'
    tf_policy_saver = PolicySaver(tf_agent.policy)
    tf_policy_saver.save(policy_dir)

if __name__ == '__main__':
    train()