import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.utils.common import function, Checkpointer
from tf_agents.environments import ParallelPyEnvironment

from env import SandShapingEnv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm.auto import tqdm, trange

from tf_agents.system.system_multiprocessing import handle_main
import os
from matplotlib import colors
from datetime import datetime

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

def train(num_parallel_envs=8, vis_interval=1000, eval_interval=1000, checkpoint_interval=10000, seed=None, batch_size=256, collect_steps_per_iteration=5, num_iterations=200000):
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
    conv_layer_params = ((32, 3, 2), (64, 3, 2))
    fc_layer_params = (256, 128)

    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=observation_spec, 
        output_tensor_spec=action_spec,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params
    )
    critic_net = CriticNetwork(
        input_tensor_spec=(observation_spec, action_spec),
        observation_conv_layer_params=conv_layer_params,
        action_fc_layer_params=None,
        joint_fc_layer_params=fc_layer_params,
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

    # Warm up replay buffer so we have at least one batch of data
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
            if step % checkpoint_interval == 0:
                train_checkpointer.save(global_step)
                tqdm.write(f"[Checkpoint @ {step}] train_loss = {float(train_loss):.4f}")
    except KeyboardInterrupt:
        print("Training interrupted at step", step)

    final_path = os.path.join(policy_base, 'final')
    PolicySaver(tf_agent.policy).save(final_path)


# Main entry point for multiprocessing
def main(_argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_interval', type=int, default=100,
                        help='Visualization interval')
    parser.add_argument('--num_envs', type=int, default=6,
                        help='Number of parallel environments for training')
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help='Steps between checkpoint saves')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--collect_steps', type=int, default=5, help='Number of steps to collect per iteration')
    parser.add_argument('--num_iterations', type=int, default=200000, help='Number of training iterations')
    args = parser.parse_args()
    train(vis_interval=args.vis_interval,
          num_parallel_envs=args.num_envs,
          checkpoint_interval=args.checkpoint_interval,
          seed=args.seed,
          batch_size=args.batch_size,
          collect_steps_per_iteration=args.collect_steps,
          num_iterations=args.num_iterations)

if __name__ == '__main__':
    handle_main(main)