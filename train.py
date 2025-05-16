import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.utils.common import function, Checkpointer
from tf_agents.environments import ParallelPyEnvironment

# Additional keras imports for encoder architectures
from tensorflow.keras import layers, models

from env import SandShapingEnv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm.auto import tqdm, trange

from tf_agents.system.system_multiprocessing import handle_main
import os
from matplotlib import colors
from datetime import datetime

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
    # Bottleneck
    b  = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    b  = layers.Conv2D(128, 3, padding='same', activation='relu')(b)
    # Up 1
    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(64, 3, padding='same', activation='relu')(u1)
    c3 = layers.Conv2D(64, 3, padding='same', activation='relu')(c3)
    # Up 2
    u2 = layers.UpSampling2D()(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(32, 3, padding='same', activation='relu')(u2)
    c4 = layers.Conv2D(32, 3, padding='same', activation='relu')(c4)
    flat = layers.Flatten()(c4)
    latent = layers.Dense(latent_dim, activation='relu')(flat)
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
    delta_rmses = []
    rel_improves = []
    auc_rmses = []
    slopes = []
    for _ in range(num_episodes):
        time_step = env.reset()
        # initial RMSE before any actions
        diff0 = env._env_map.difference(env._target_map)
        rmse0 = np.sqrt(np.mean(diff0**2))
        # track RMSE over time
        rmse_series = [rmse0]
        while not time_step.is_last():
            action_step = policy.action(time_step)
            action = action_step.action
            if hasattr(action, "numpy"):
                action = action.numpy()
            time_step = env.step(action)
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
    num_parallel_envs=8,
    vis_interval=1000,
    eval_interval=1000,
    checkpoint_interval=10000,
    seed=None,
    batch_size=256,
    collect_steps_per_iteration=5,
    num_iterations=200000,
    use_heuristic_warmup: bool = False,
    encoder_type: str = 'cnn'
):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    # Hyperparameters
    replay_buffer_capacity = 16384
    learning_rate = 3e-4
    gamma = 0.99
    num_eval_episodes = 5

    # Create seeded Python environments for training, evaluation, and visualization
    env_fns = []
    for idx in range(num_parallel_envs):
        env_fns.append(lambda idx=idx: SandShapingEnv(seed=(seed + idx) if seed is not None else None))
    train_py_env = ParallelPyEnvironment(env_fns)
    eval_py_env = SandShapingEnv(seed=seed)
    vis_env = SandShapingEnv(seed=(seed + num_parallel_envs) if seed is not None else None)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Network architecture
    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    # Choose encoder architecture
    if encoder_type == 'cnn':
        actor_conv_params = ((32, 3, 2), (64, 3, 2))
        critic_conv_params = actor_conv_params
        actor_preproc = None
        critic_preproc = None
    elif encoder_type == 'unet':
        actor_conv_params = None
        critic_conv_params = None
        # Build separate UNet encoders for actor and critic to avoid layer-copy warnings
        actor_preproc = build_unet_encoder(observation_spec.shape)
        critic_preproc = build_unet_encoder(observation_spec.shape)
    else:
        raise ValueError(f"Unknown encoder_type '{encoder_type}'.")

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

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=num_parallel_envs,
        max_length=replay_buffer_capacity
    )
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2,
        single_deterministic_pass=False
    ).prefetch(2)
    iterator = iter(dataset)

    collect_driver = DynamicStepDriver(
        train_env,
        tf_agent.collect_policy,
        observers=[replay_buffer.add_batch],
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
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--collect_steps', type=int, default=4, help='Number of steps to collect per iteration')
    parser.add_argument('--checkpoint_interval', type=int, default=0, help='Steps between checkpoint saves')
    parser.add_argument('--eval_interval', type=int, default=0, help='Steps between evaluation')
    parser.add_argument('--vis_interval', type=int, default=0, help='Visualization interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--heuristic_warmup', action='store_true', default=True, help='Use heuristic policy for warm-up instead of random actions')
    parser.add_argument('--encoder', type=str, default='unet', choices=['cnn', 'unet'], help='Backbone encoder to use for actor/critic')

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
          encoder_type=args.encoder)

if __name__ == '__main__':
    handle_main(main)