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
from tensorflow.keras.layers import Conv2D, Resizing

from env import SandShapingEnv
import matplotlib.pyplot as plt
import numpy as np

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
    batch_size = 64
    learning_rate = 3e-4
    gamma = 0.99
    eval_interval = 10000
    num_eval_episodes = 5
    # Visualization settings
    vis_interval = 10

    # Create environments
    train_py_env = SandShapingEnv()
    eval_py_env = SandShapingEnv()
    # For fixed colormap scaling in visualization
    max_amp = train_py_env._amplitude_range[1]
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Define networks
    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        conv_layer_params=((16, 3, 2),      # 16 filters, 3×3 kernel, stride 2
                        (32, 3, 2)),        # 32 filters, 3×3 kernel, stride 2
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
        if step % vis_interval == 0:
            print(f'step={step}: loss={train_loss.numpy():.4f}')

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print(f'step={step}: avg_return={avg_return:.2f}')


        # Non-blocking Matplotlib visualization every vis_interval steps
        if step % vis_interval == 0:
            # Mean-center env and target maps
            env_raw = train_py_env._env_map.map
            target_raw = train_py_env._target_map.map
            env_img = env_raw - np.mean(env_raw)
            target_img = target_raw - np.mean(target_raw)
            diff_img, reward = train_py_env._env_map.compute_reward(train_py_env._target_map)
            print(reward)
            axes_vis[0].clear()
            axes_vis[0].imshow(env_img, cmap='turbo')
            axes_vis[0].set_title(f'Env @ step {step}')
            axes_vis[1].clear()
            axes_vis[1].imshow(target_img, cmap='turbo')
            axes_vis[1].set_title('Target')
            axes_vis[2].clear()
            axes_vis[2].imshow(diff_img, cmap='turbo', vmin=-max_amp, vmax=max_amp)
            axes_vis[2].set_title('Difference')
            fig_vis.canvas.draw()
            plt.pause(0.001)

    # Save final policy
    policy_dir = 'policy'
    tf_policy_saver = PolicySaver(tf_agent.policy)
    tf_policy_saver.save(policy_dir)

if __name__ == '__main__':
    train()