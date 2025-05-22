import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.utils.common import function, Checkpointer
from tf_agents.environments import ParallelPyEnvironment

# Additional keras imports for encoder architectures
from keras import layers, models

import tensorflow_probability as tfp
from tf_agents.networks import network

from env import SandShapingEnv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm.auto import tqdm, trange

from tf_agents.system.system_multiprocessing import handle_main
import os
from matplotlib import colors
from datetime import datetime

# ==================== FPN/CoordConv/Custom Actor & Critic ====================
class CoordConv(layers.Layer):
    def __init__(self, filters=32, kernel_size=1, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
        self.coords = None

    def build(self, input_shape):
        # Precompute coordinate grid [1, H, W, 2]
        batch_dim, h, w, _ = input_shape
        xx = tf.linspace(-1.0, 1.0, w)
        yy = tf.linspace(-1.0, 1.0, h)
        xx = tf.reshape(xx, [1, 1, w])
        xx = tf.tile(xx, [1, h, 1])
        yy = tf.reshape(yy, [1, h, 1])
        yy = tf.tile(yy, [1, 1, w])
        self.coords = tf.stack([xx, yy], axis=-1)  # shape [1,h,w,2]
        super().build(input_shape)

    def call(self, x):
        # x is expected to be [B, H, W, C]
        batch_size = tf.shape(x)[0]
        coords = tf.tile(self.coords, [batch_size, 1, 1, 1])
        conv_input = tf.concat([x, coords], axis=-1)
        return self.conv(conv_input)

class FPNBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.shortcut = None
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.bn1 = layers.LayerNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.LayerNormalization()
        self.relu = layers.Activation('relu')

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels != self.filters:
            self.shortcut = layers.Conv2D(self.filters, 1, padding='same')
        super().build(input_shape)

    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.shortcut is not None:
            x_proj = self.shortcut(x)
        else:
            x_proj = x
        out = self.relu(x_proj + y)
        return out

class FPNEncoder(layers.Layer):
    def __init__(self, filters_list=(32, 64, 128), latent_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.coordconv = CoordConv()
        self.downs = []
        for f in filters_list:
            self.downs.append(FPNBlock(f))
        self.pools = [layers.MaxPool2D() for _ in filters_list]
        # FPN lateral and upsample layers
        fpn_channels = filters_list[-1]
        self.lateral_convs = [layers.Conv2D(fpn_channels, 1, padding='same') for _ in filters_list]
        self.upsamples     = [layers.UpSampling2D(size=2) for _ in filters_list[:-1]]
        self.merge_upsamples = [layers.UpSampling2D(size=2**i) for i in range(len(filters_list))]
        self.global_pool = layers.GlobalAveragePooling2D()
        self.latent = layers.Dense(latent_dim, activation='relu')
    def call(self, x):
        # Bottom-up pass
        x = self.coordconv(x)
        c_feats = []
        for block, pool in zip(self.downs, self.pools):
            x = block(x)
            c_feats.append(x)
            x = pool(x)
        # Top-down lateral fusion
        p_levels = [None] * len(c_feats)
        last = len(c_feats) - 1
        p_levels[last] = self.lateral_convs[last](c_feats[last])
        for i in range(last - 1, -1, -1):
            p_levels[i] = self.lateral_convs[i](c_feats[i]) + self.upsamples[i](p_levels[i+1])
        # Merge multi-scale features
        merged = tf.concat([self.merge_upsamples[i](p_levels[i]) for i in range(len(p_levels))], axis=-1)
        x = self.global_pool(merged)
        return self.latent(x)
    
class SpatialSoftmax(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, logits):
        # logits: [B,H,W,1]
        # Extract batch and spatial dims without Python iteration
        shape_un = tf.unstack(tf.shape(logits))
        b = shape_un[0]
        h = shape_un[1]
        w = shape_un[2]
        flat = tf.reshape(logits, [b, h * w])
        prob = tf.nn.softmax(flat)
        coords_x, coords_y = tf.meshgrid(
            tf.linspace(0.0, 1.0, w), tf.linspace(0.0, 1.0, h)
        )
        coords = tf.stack([tf.reshape(coords_x, [-1]), tf.reshape(coords_y, [-1])], axis=1)
        exp = tf.matmul(prob, coords)
        return exp  # [B,2]

class CarveActorNetwork(network.Network):
    def __init__(self, observation_spec, action_spec, name='CarveActorNetwork'):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        self._action_spec = action_spec
        self.encoder = FPNEncoder(latent_dim=128)
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        # Output mean/logstd for each action dimension
        self.mean = layers.Dense(action_spec.shape[0])
        self.logstd = layers.Dense(action_spec.shape[0])
    def call(self, observations, step_type=None, network_state=(), training=False):
        x = tf.cast(observations, tf.float32)
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.fc2(x)
        mean = self.mean(x)
        logstd = self.logstd(x)
        # Prevent extreme log-std values and enforce minimum scale
        logstd = tf.clip_by_value(logstd, -3.0, 1.0)
        std = tf.nn.softplus(logstd) + 1e-3

        # Base Gaussian distribution
        base_dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
        # Squash with tanh to [-1,1], then scale to [–0.5,0.5] and shift to [0,1]
        bijector = tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=0.5),
            tfp.bijectors.Scale(scale=0.5),
            tfp.bijectors.Tanh()
        ])
        dist = tfp.distributions.TransformedDistribution(distribution=base_dist, bijector=bijector)
        return dist, network_state

class CarveCriticNetwork(network.Network):
    def __init__(self, observation_spec, action_spec, name='CarveCriticNetwork'):
        super().__init__(input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)
        self.encoder = FPNEncoder(latent_dim=128)
        self.action_fc = layers.Dense(64, activation='relu')
        self.concat_fc1 = layers.Dense(128, activation='relu')
        self.concat_fc2 = layers.Dense(64, activation='relu')
        self.q_out = layers.Dense(1)
        # Gaussian attention parameters
        self._gauss_sigma = 0.05  # width of Gaussian bump in normalized coords
        self._grid = None

    def build(self, input_shape):
        # input_shape: tuple(obs_shape, action_shape)
        obs_shape, _ = input_shape
        # obs_shape: (batch, H, W, C)
        _, H, W, _ = obs_shape
        # Create normalized coordinate grid [0,1] for x and y
        ys = tf.linspace(0.0, 1.0, H)
        xs = tf.linspace(0.0, 1.0, W)
        grid_y, grid_x = tf.meshgrid(ys, xs, indexing='ij')  # shape [H, W]
        grid = tf.stack([grid_x, grid_y], axis=-1)          # [H, W, 2]
        grid = tf.reshape(grid, [1, H, W, 2])                # [1, H, W, 2]
        self._grid = tf.cast(grid, tf.float32)
        super().build(input_shape)

    def call(self, inputs, step_type=None, network_state=(), training=False):
        obs, actions = inputs
        # Cast observation
        obs = tf.cast(obs, tf.float32)  # [B, H, W, C]
        # Gaussian attention map from action (x,y)
        # Extract normalized x,y from actions
        xy = actions[..., :2]  # [B, 2]
        # Reshape to [B, 1, 1, 2]
        xy = tf.reshape(xy, [-1, 1, 1, 2])
        # Compute squared distance on grid (broadcasting grid over batch)
        # self._grid: [1, H, W, 2]
        dist2 = tf.reduce_sum((self._grid - xy)**2, axis=-1, keepdims=True)  # [B, H, W, 1]
        gauss = tf.exp(-dist2 / (2 * self._gauss_sigma**2))                   # [B, H, W, 1]
        # Augment observation with Gaussian bump channel
        obs_aug = tf.concat([obs, gauss], axis=-1)  # [B, H, W, C+1]
        # Encode augmented observation
        x = self.encoder(obs_aug)
        # Process action latents
        a = self.action_fc(actions)
        x = tf.concat([x, a], axis=-1)
        x = self.concat_fc1(x)
        x = self.concat_fc2(x)
        q = self.q_out(x)
        return tf.squeeze(q, axis=-1), network_state

# --- Heuristic policy imports ---
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step

# --- Visualization function ---
def visualize(fig, axes, cbars, env, step, max_amp):
    """
    Update the visual debugging plots for the given environment and step.
    """
    # Extract maps and observation
    h = env._env_map.map
    t = env._target_map.map
    diff = env._env_map.difference(env._target_map)
    obs = env._build_observation(diff, h, t)
    # Prepare value ranges
    hmin, hmax = h.min(), h.max()
    tmin, tmax = t.min(), t.max()
    dlim = max_amp / 2
    norm_diff = colors.TwoSlopeNorm(vcenter=0, vmin=-dlim, vmax=dlim)
    # Data, colormaps, and titles for subplots
    channels = [diff, h, t, obs[..., 0], obs[..., 1], obs[..., 2]]
    cmaps = ['turbo', 'viridis', 'viridis', 'turbo', 'viridis', 'viridis']
    titles = [
        f'Diff raw\nmin:{diff.min():.1f}, max:{diff.max():.1f}, rmse={np.sqrt(np.mean(diff**2)):.1f}',
        f'Env height @ step {step}\nmin:{hmin:.1f}, max:{hmax:.1f}',
        f'Target height\nmin:{tmin:.1f}, max:{tmax:.1f}',
        'Diff channel',
        'Env channel',
        'Tgt channel',
    ]
    # Update each subplot
    for idx, ax in enumerate(axes.flat):
        ax.clear()
        data = channels[idx]
        cmap = cmaps[idx]
        if idx == 0:
            im = ax.imshow(data, cmap=cmap, norm=norm_diff)
        else:
            im = ax.imshow(data, cmap=cmap)
        ax.set_title(titles[idx])
        # Colorbar handling
        if cbars[idx] is None:
            cbars[idx] = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            cbars[idx].mappable = im
            cbars[idx].update_normal(im)
    fig.tight_layout()


# ---------- UNet Encoder ----------------------------------------------------
def build_unet_encoder(input_shape, latent_dim=256):
    """
    Returns a keras.Model that maps an (H,W,C) observation to a latent vector
    of length `latent_dim`.  A lightweight 2‑down 2‑up UNet.
    """
    inputs = layers.Input(shape=input_shape)
    # Down 1
    c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPool2D()(c1)
    # Down 2
    c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPool2D()(c2)
    # Down 3
    c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPool2D()(c3)
    # Bottleneck
    b  = layers.Conv2D(128, 3, padding='same', activation='relu')(p3)
    b  = layers.Conv2D(128, 3, padding='same', activation='relu')(b)
    # Up 1
    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c3])
    c4 = layers.Conv2D(128, 3, padding='same', activation='relu')(u1)
    c4 = layers.Conv2D(128, 3, padding='same', activation='relu')(c4)
    # Up 2
    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)
    c5 = layers.Conv2D(64, 3, padding='same', activation='relu')(c5)
    # Up 3
    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c1])
    c6 = layers.Conv2D(32, 3, padding='same', activation='relu')(u3)
    c6 = layers.Conv2D(32, 3, padding='same', activation='relu')(c6)
    pooled = layers.GlobalAveragePooling2D()(c6)
    latent = layers.Dense(latent_dim, activation='relu')(pooled)
    return models.Model(inputs, latent, name='unet_encoder')


# ---------- Heuristic Policy --------------------------------------------------
class HeuristicPressPolicy(py_policy.PyPolicy):
    """
    Greedy one‑step policy: press at the (x,y) with maximum positive
    env‑minus‑target height difference.  Uses only the diff channel that is
    already present in the observation, so it works with ParallelPyEnvironment.
    """
    def __init__(self, time_step_spec, action_spec,
                 width, height, tool_radius, amp_max):
        super().__init__(time_step_spec, action_spec)
        self._w = width
        self._h = height
        self._r = tool_radius
        self._amp_max = amp_max        # same as env._amplitude_range[1]
        self._inv_depth = 1.0 / (0.66 * tool_radius)   # for dz normalisation

    # ----- utility -----------------------------------------------------------
    def _single_action(self, diff_signed):
        """
        diff_signed: (H,W) array, range [-1,1]  (already clipped)
        """
        cy, cx = np.unravel_index(np.argmax(diff_signed), diff_signed.shape)

        # (x,y) → normalised action coords
        x_norm = (cx - self._r) / max(1e-6, (self._w - 2 * self._r))
        y_norm = (cy - self._r) / max(1e-6, (self._h - 2 * self._r))
        x_norm = float(np.clip(x_norm, 0.0, 1.0))
        y_norm = float(np.clip(y_norm, 0.0, 1.0))

        # Estimate absolute diff height from signed channel
        # diff_signed = diff * (2/amp_max)  ⇒  diff ≈ diff_signed * amp_max/2
        diff_abs = diff_signed[cy, cx] * (self._amp_max * 0.5)
        depth = max(0.0, diff_abs * 1.1)        # 10 % overshoot
        depth = min(depth, 0.66 * self._r)      # respect env max depth
        dz_norm = depth * self._inv_depth       # scale to [0,1]
        dz_norm = float(np.clip(dz_norm, 0.0, 1.0))

        return np.array([x_norm, y_norm, dz_norm], dtype=np.float32)

    # ----- PyPolicy overrides -------------------------------------------------
    def _action(self, time_step, policy_state):
        obs = time_step.observation           # (B,H,W,3) or (H,W,3)
        if obs.ndim == 4:                     # batched
            batch_actions = [self._single_action(obs[i, ..., 0])
                             for i in range(obs.shape[0])]
            act = np.stack(batch_actions, axis=0)
        else:
            act = self._single_action(obs[..., 0])
        return policy_step.PolicyStep(act, policy_state, ())


def compute_eval(env, policy, num_episodes=10):
    """
    Evaluate error-focused metrics over multiple episodes.
    Returns a dict containing:
      - init_rmse_mean, final_rmse_mean, delta_rmse_mean, rel_improve_mean, auc_rmse_mean, slope_rmse_mean
      - and per-episode lists for each metric.
    """
    # Wrap raw env for policy calls
    tf_env = tf_py_environment.TFPyEnvironment(env)
    delta_rmses = []
    rel_improves = []
    auc_rmses = []
    slopes = []
    for _ in range(num_episodes):
        time_step = tf_env.reset()
        # initial RMSE before any actions
        diff0 = env._env_map.difference(env._target_map)
        rmse0 = np.sqrt(np.mean(diff0**2))
        # track RMSE over time
        rmse_series = [rmse0]
        while not time_step.is_last():
            action_step = policy.action(time_step)
            batched_action = action_step.action  # already shape [1, action_dim]
            time_step = tf_env.step(batched_action)
            # Underlying env state is the wrapped `env`
            diff = env._env_map.difference(env._target_map)
            rmse_series.append(np.sqrt(np.mean(diff**2)))
        # compute per-episode metrics
        rmse_initial = rmse_series[0]
        rmse_final = rmse_series[-1]
        delta = rmse_initial - rmse_final
        rel = delta / rmse_initial if rmse_initial > 0 else 0.0
        auc = np.trapz(rmse_series)
        steps = np.arange(len(rmse_series))
        slope = np.polyfit(steps, rmse_series, 1)[0]
        delta_rmses.append(delta)
        rel_improves.append(rel)
        auc_rmses.append(auc)
        slopes.append(slope)
    # aggregate means
    metrics = {
        'delta_rmse_mean': float(np.mean(delta_rmses)),
        'rel_improve_mean': float(np.mean(rel_improves)),
        'auc_rmse_mean': float(np.mean(auc_rmses)),
        'slope_rmse_mean': float(np.mean(slopes)),
        'delta_rmse_list': delta_rmses,
        'rel_improve_list': rel_improves,
        'auc_rmse_list': auc_rmses,
        'slope_rmse_list': slopes,
    }
    return metrics

def train(
    num_parallel_envs: int = 4,
    vis_interval: int = 1000,
    eval_interval: int = 1000,
    checkpoint_interval: int = 10000,
    seed: int | None = None,
    batch_size: int = 64,
    collect_steps_per_iteration: int = 4,
    num_iterations: int = 200000,
    use_heuristic_warmup: bool = False,
    encoder_type: str = 'cnn',
    debug: bool = False
):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    # Hyperparameters
    replay_buffer_capacity = max(4096, batch_size * 64)
    learning_rate = 1e-4
    gamma = 0.99
    num_eval_episodes = 5

    # Create seeded Python environments for training, evaluation, and visualization
    env_fns = []
    for idx in range(num_parallel_envs):
        # enable debug mode in each Python env
        env_fns.append(lambda idx=idx: SandShapingEnv(debug=True,
                                                      seed=(seed + idx) if seed is not None else None))
    train_py_env = ParallelPyEnvironment(env_fns)
    # expose per-env debug scalars later
    python_envs = train_py_env._envs  # list of SandShapingEnv instances
    eval_py_env = SandShapingEnv(seed=seed)
    vis_env = SandShapingEnv(seed=(seed + num_parallel_envs) if seed is not None else None)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Network architecture
    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    # Choose encoder architecture
    if encoder_type == 'cnn':
        actor_conv_params = ((32, 3, 2), (64, 3, 2), (128, 3, 2))
        critic_conv_params = actor_conv_params
        actor_preproc = None
        critic_preproc = None
    elif encoder_type == 'unet':
        actor_conv_params = None
        critic_conv_params = None
        actor_preproc = build_unet_encoder(observation_spec.shape)
        critic_preproc = build_unet_encoder(observation_spec.shape)
    elif encoder_type == 'fpn':
        actor_net = CarveActorNetwork(observation_spec, action_spec)
        critic_net = CarveCriticNetwork(observation_spec, action_spec)
    else:
        raise ValueError(f"Unknown encoder_type '{encoder_type}'.")

    if encoder_type == 'fpn':
        # networks already set
        pass
    else:
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            input_tensor_spec=observation_spec,
            output_tensor_spec=action_spec,
            preprocessing_layers=actor_preproc,
            conv_layer_params=actor_conv_params,
            fc_layer_params=(256, 128) if encoder_type == 'cnn' else (128,),
        )
        critic_net = CriticNetwork(
            input_tensor_spec=(observation_spec, action_spec),
            observation_conv_layer_params=critic_conv_params,
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=(256, 128) if encoder_type == 'cnn' else (128,),
            name='critic_network'
        )

    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf_agent = sac_agent.SacAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate, clipnorm=1.0),
        critic_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate, clipnorm=1.0),
        alpha_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate, clipnorm=1.0),
        target_update_tau=0.005,
        target_update_period=1,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=1.0,
        train_step_counter=global_step
    )
    tf_agent.initialize()

    checkpoint_dir = 'ckpts'
    policy_base    = 'policies'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(policy_base, exist_ok=True)
    policy_saver = PolicySaver(tf_agent.policy)
    # ---- logging setup ----
    log_root = 'logs'
    os.makedirs(log_root, exist_ok=True)
    logdir = os.path.join(log_root, datetime.now().strftime('%Y%m%d-%H%M%S'))
    summary_writer = tf.summary.create_file_writer(logdir)
    summary_writer.set_as_default()

    from tf_agents.trajectories import trajectory as traj_lib

    def collect_summary(trajectory):
        # Log the mean reward of the collected trajectories
        tf.summary.scalar('train/step_reward', tf.reduce_mean(trajectory.reward), step=global_step)

    def action_summary(trajectory):
        # Log mean action components across the batch
        actions = trajectory.action
        tf.summary.scalar('train/action_x_mean',
                          tf.reduce_mean(actions[..., 0]), step=global_step)
        tf.summary.scalar('train/action_y_mean',
                          tf.reduce_mean(actions[..., 1]), step=global_step)
        tf.summary.scalar('train/action_depth_mean',
                          tf.reduce_mean(actions[..., 2]), step=global_step)

    def buffer_summary(trajectory):
        # Log current size of the replay buffer
        tf.summary.scalar('replay_buffer/size',
                          replay_buffer.num_frames(), step=global_step)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=num_parallel_envs,
        max_length=replay_buffer_capacity
    )
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2,
        single_deterministic_pass=False
    ).prefetch(tf.data.AUTOTUNE)
    iterator = iter(dataset)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        tf_agent.collect_policy,
        observers=[
            replay_buffer.add_batch,
            collect_summary,
            action_summary,
            buffer_summary
        ],
        num_steps=collect_steps_per_iteration
    )

    # Warm-up buffer: heuristic or random
    if use_heuristic_warmup:
        # Python-driver heuristic warm-up on the parallel Python env
        HEURISTIC_WARMUP_FRAMES = max(batch_size * 10, 5000)
        # Initialize warm-up timestep:
        warmup_ts = train_py_env.reset()
        # Use a single eval env for shape parameters
        warmup_env = eval_py_env
        heuristic_policy = HeuristicPressPolicy(
            time_step_spec=train_py_env.time_step_spec(),
            action_spec=action_spec,
            width=warmup_env._width,
            height=warmup_env._height,
            tool_radius=warmup_env._tool_radius,
            amp_max=warmup_env._amplitude_range[1],
        )
        heuristic_driver = PyDriver(
            train_py_env,
            heuristic_policy,
            observers=[replay_buffer.add_batch],
            max_steps=collect_steps_per_iteration
        )
        while replay_buffer.num_frames() < HEURISTIC_WARMUP_FRAMES:
            warmup_ts, _ = heuristic_driver.run(warmup_ts)
    else:
        # Random warm-up to at least one batch
        while replay_buffer.num_frames() < batch_size:
            collect_driver.run()

    train_checkpointer = Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=3,
        agent=tf_agent,
        policy=tf_agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    train_checkpointer.initialize_or_restore()

    if vis_interval > 0:
        fig_vis, axes_vis = plt.subplots(2, 3, figsize=(12, 9))
        cbars = [None] * 6  # one colorbar placeholder per subplot
        max_amp = vis_env._amplitude_range[1]
        vis_ts = vis_env.reset()

    @function
    def train_step():
        collect_driver.run()
        experience, _ = next(iterator)
        return tf_agent.train(experience)

    try:
        for step in trange(1, num_iterations + 1, desc='Training'):
            train_info = train_step()
            train_loss = train_info.loss
            if debug:
                # log per-step debug scalars to TensorBoard (only in debug single-env mode)
                step_removed = [env._last_removed_norm for env in python_envs]
                step_rel     = [env._last_rel_improve for env in python_envs]
                step_reward  = [env._last_reward for env in python_envs]
                tf.summary.scalar('train/removed_volume_norm',
                                  float(np.mean(step_removed)), step=step)
                tf.summary.scalar('train/rel_improve',
                                  float(np.mean(step_rel)), step=step)
                tf.summary.scalar('train/step_reward',
                                  float(np.mean(step_reward)), step=step)
            if vis_interval > 0 and step % vis_interval == 0:
                if vis_ts.is_last():
                    vis_ts = vis_env.reset()
                action_vis = tf_agent.policy.action(vis_ts).action
                if hasattr(action_vis, "numpy"): action_vis = action_vis.numpy()
                vis_ts = vis_env.step(action_vis)
                visualize(fig_vis, axes_vis, cbars, vis_env, step, max_amp)
                # Log map images for TensorBoard
                diff_norm = (axes_vis.flat[0].images[0].get_array() - (-1)) / 2.0
                tf.summary.image('maps/diff', diff_norm[np.newaxis, ..., np.newaxis], step=step)
                obs_env = axes_vis.flat[4].images[0].get_array()
                obs_tgt = axes_vis.flat[5].images[0].get_array()
                tf.summary.image('maps/env', obs_env[np.newaxis, ..., np.newaxis], step=step)
                tf.summary.image('maps/target', obs_tgt[np.newaxis, ..., np.newaxis], step=step)
            if eval_interval > 0 and step % eval_interval == 0:
                # Detailed evaluation metrics
                metrics = compute_eval(eval_py_env, tf_agent.policy, num_eval_episodes)
                # Log error-focused metrics to TensorBoard
                tf.summary.scalar('eval/delta_rmse_mean', metrics['delta_rmse_mean'], step=step)
                tf.summary.scalar('eval/rel_improve_mean', metrics['rel_improve_mean'], step=step)
                tf.summary.scalar('eval/auc_rmse_mean',   metrics['auc_rmse_mean'],   step=step)
                tf.summary.scalar('eval/slope_rmse_mean', metrics['slope_rmse_mean'], step=step)
                # Console output for quick look
                tqdm.write(
                    f"[Eval @ {step}] ΔRMSE: {metrics['delta_rmse_mean']:.4f}, "
                    f"RelImprove: {metrics['rel_improve_mean']:.2%}, "
                )
            if checkpoint_interval > 0 and step % checkpoint_interval == 0:
                train_checkpointer.save(global_step)
                tqdm.write(f"[Checkpoint @ {step}] train_loss = {float(train_loss):.4f}")
    except KeyboardInterrupt:
        print("Training interrupted at step", step)

    final_path = os.path.join(policy_base, 'final')
    PolicySaver(tf_agent.policy).save(final_path)


# Main entry point for multiprocessing
def main(_argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=200000, help='Number of training iterations')
    parser.add_argument('--num_envs', type=int, default=6, help='Number of parallel environments for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--collect_steps', type=int, default=8, help='Number of steps to collect per iteration')
    parser.add_argument('--checkpoint_interval', type=int, default=0, help='Steps between checkpoint saves')
    parser.add_argument('--eval_interval', type=int, default=5000, help='Steps between evaluation')
    parser.add_argument('--vis_interval', type=int, default=0, help='Visualization interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--heuristic_warmup', action='store_true', default=True, help='Use heuristic policy for warm-up instead of random actions')
    parser.add_argument('--encoder', type=str, default='cnn', choices=['cnn', 'unet', 'fpn'], help='Backbone encoder to use for actor/critic')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug-mode scalar logging for single-process runs')

    args = parser.parse_args()
    
    train(vis_interval=args.vis_interval,
          eval_interval=args.eval_interval,
          num_parallel_envs=args.num_envs,
          checkpoint_interval=args.checkpoint_interval,
          seed=args.seed,
          batch_size=args.batch_size,
          collect_steps_per_iteration=args.collect_steps,
          num_iterations=args.num_iterations,
          use_heuristic_warmup=args.heuristic_warmup,
          encoder_type=args.encoder,
          debug=args.debug)

if __name__ == '__main__':
    handle_main(main)