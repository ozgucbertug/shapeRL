import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.metrics import tf_metrics
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

def sample_random_action(spec):
    return np.random.uniform(low=spec.minimum,
                             high=spec.maximum).astype(np.float32)

def compute_avg_return(environment, policy, num_episodes=10):
    """Run a fixed number of episodes in `environment` using `policy`
    and return the *Python float* average return.
    """
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action = policy.action(time_step).action
            time_step = environment.step(action)
            # Cast to float to detach from TF / NumPy
            episode_return += float(time_step.reward)
        total_return += episode_return
    return total_return / num_episodes

def train(vis_interval=50, num_parallel_envs=8, log_interval=100):
    # Hyperparameters
    num_iterations = 200000
    collect_steps_per_iteration = 5
    replay_buffer_capacity = 2048
    batch_size = 32
    learning_rate = 3e-4
    gamma = 0.99
    eval_interval = 20000
    num_eval_episodes = 5
    warmup_batches = batch_size // num_parallel_envs



    def make_env():
        return SandShapingEnv()
    train_py_env = ParallelPyEnvironment([make_env for _ in range(num_parallel_envs)])
    eval_py_env  = SandShapingEnv()

    vis_env = SandShapingEnv()
    max_amp = vis_env._amplitude_range[1]

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

    train_loss_hist = []
    eval_return_hist = []

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

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),                 # no batch_size arg
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=num_parallel_envs),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=num_parallel_envs),
    ]

    collect_driver = DynamicStepDriver(
        train_env,
        tf_agent.collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=collect_steps_per_iteration
    )

    train_checkpointer = Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=tf_agent,
        policy=tf_agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    train_checkpointer.initialize_or_restore()

    for _ in range(warmup_batches):
        collect_driver.run()

    if vis_interval > 0:
        fig_vis, axes_vis = plt.subplots(2, 3, figsize=(12, 9))
        cbars = [None] * 6  # one colorbar placeholder per subplot

    @function
    def train_step():
        collect_driver.run()
        experience, _ = next(iterator)
        return tf_agent.train(experience)

    for step in trange(1, num_iterations + 1, desc='Training'):
        train_info = train_step()
        train_loss = train_info.loss
        vis_env.step(sample_random_action(vis_env.action_spec()))
        if step % log_interval == 0:
            loss_val = float(train_loss)
            tqdm.write(f'step {step}: train_loss = {loss_val:.4f}')
            train_loss_hist.append((step, loss_val))
            tf.summary.scalar('train/loss', loss_val, step=step)
            # Log SAC internals
            tf.summary.scalar('train/critic_loss', train_info.extra.critic_loss, step=step)
            tf.summary.scalar('train/actor_loss',  train_info.extra.actor_loss,  step=step)
            tf.summary.scalar('train/alpha_loss',  train_info.extra.alpha_loss,  step=step)
            # Log replay buffer occupancy
            tf.summary.scalar('replay/size', replay_buffer.num_frames(), step=step)
            for m in train_metrics:
                tf.summary.scalar(f'train/{m.name}', m.result(), step=step)
                m.reset()
        if vis_interval > 0 and step % vis_interval == 0:
            # Compute maps and observation
            h = vis_env._env_map.map
            t = vis_env._target_map.map
            diff = vis_env._env_map.difference(vis_env._target_map)
            obs = vis_env._build_observation(diff, h, t)

            # Prepare limits and norms
            hmin, hmax = h.min(), h.max()
            tmin, tmax = t.min(), t.max()
            dlim = max_amp / 2
            norm_diff = colors.TwoSlopeNorm(vcenter=0, vmin=-dlim, vmax=dlim)

            # First row: raw maps
            # Diff raw (col 0)
            ax0 = axes_vis[0, 0]; ax0.clear()
            im0 = ax0.imshow(diff, cmap='turbo', norm=norm_diff)
            ax0.set_title(f'Diff raw\nmin:{diff.min():.1f}, max:{diff.max():.1f}, rmse={np.sqrt(np.sum(diff**2)):.1f}')

            # Env height raw (col 1)
            ax1 = axes_vis[0, 1]; ax1.clear()
            im1 = ax1.imshow(h, cmap='viridis')
            ax1.set_title(f'Env height @ step {step}\nmin:{hmin:.1f}, max:{hmax:.1f}')

            # Target height raw (col 2)
            ax2 = axes_vis[0, 2]; ax2.clear()
            im2 = ax2.imshow(t, cmap='viridis')
            ax2.set_title(f'Target height\nmin:{tmin:.1f}, max:{tmax:.1f}')

            # Second row: observation channels
            # Difference channel
            ax = axes_vis[1, 0]; ax.clear()
            im3 = ax.imshow(obs[..., 0], cmap='turbo')
            ax.set_title(f'Difference channel\n(step {step})')

            # Current height channel
            ax = axes_vis[1, 1]; ax.clear()
            im4 = ax.imshow(obs[..., 1], cmap='viridis')
            ax.set_title(f'Current height channel\n(step {step})')

            # Target height channel
            ax = axes_vis[1, 2]; ax.clear()
            im5 = ax.imshow(obs[..., 2], cmap='viridis')
            ax.set_title(f'Target height channel\n(step {step})')

            # Create or update one colorbar per subplot (avoid duplicates)
            for idx, (ax, im) in enumerate(zip(axes_vis.flat, [im0, im1, im2, im3, im4, im5])):
                if cbars[idx] is None:
                    cbars[idx] = fig_vis.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    cbars[idx].mappable = im
                    cbars[idx].update_normal(im)
            fig_vis.tight_layout()

            # Log map images for visual debugging
            tf.summary.image('maps/diff',   diff[np.newaxis, ..., np.newaxis], step=step)
            tf.summary.image('maps/env',    h  [np.newaxis, ..., np.newaxis], step=step)
            tf.summary.image('maps/target', t  [np.newaxis, ..., np.newaxis], step=step)

            if plt.get_fignums():
                fig_vis.canvas.draw()
            plt.pause(0.001)
        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            tqdm.write(f'step={step}: avg_return={float(avg_return):.2f}')
            eval_return_hist.append((step, float(avg_return)))
            tf.summary.scalar('eval/avg_return', avg_return, step=step)
            train_checkpointer.save(global_step)
            tqdm.write(f'Checkpoint saved at step {step}')

    # ---- save & plot logged metrics ----
    metrics_path = os.path.join(logdir, 'metrics.npz')
    np.savez(metrics_path,
             steps_loss=np.array([s for s, _ in train_loss_hist]),
             loss=np.array([v for _, v in train_loss_hist]),
             steps_eval=np.array([s for s, _ in eval_return_hist]),
             avg_return=np.array([v for _, v in eval_return_hist]))

    if train_loss_hist and eval_return_hist:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        steps_loss = [s for s, _ in train_loss_hist]
        loss_vals = [v for _, v in train_loss_hist]
        ax1.plot(steps_loss, loss_vals, label='Train loss')
        ax1.set_xlabel('Environment steps')
        ax1.set_ylabel('Train loss')

        ax2 = ax1.twinx()
        steps_eval = [s for s, _ in eval_return_hist]
        ret_vals = [v for _, v in eval_return_hist]
        ax2.plot(steps_eval, ret_vals, linestyle='--', label='Avg return')
        ax2.set_ylabel('Average return')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)
        fig.tight_layout()
        fig.savefig(os.path.join(logdir, 'metrics.png'))
        plt.close(fig)

    final_path = os.path.join(policy_base, 'final')
    PolicySaver(tf_agent.policy).save(final_path)


# Main entry point for multiprocessing
def main(_argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_interval', type=int, default=100,
                        help='Visualization interval')
    parser.add_argument('--num_envs', type=int, default=6,
                        help='Number of parallel environments for training')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval in environment steps')
    args = parser.parse_args()
    train(vis_interval=args.vis_interval,
          num_parallel_envs=args.num_envs,
          log_interval=args.log_interval)

if __name__ == '__main__':
    handle_main(main)