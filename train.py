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
from tf_agents.environments import parallel_py_environment

from env import SandShapingEnv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm.auto import tqdm, trange

# Add multiprocessing import
from tf_agents.system import multiprocessing as tf_mp
import os
import json
from pathlib import Path

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

def train(config_path='config.json'):
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)
    else:
        raise FileNotFoundError(f"Config file {cfg_file} not found.")

    num_iterations = cfg["num_iterations"]
    collect_steps_per_iteration = cfg["collect_steps_per_iteration"]
    replay_buffer_capacity = cfg["replay_buffer_capacity"]
    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    gamma = cfg["gamma"]
    eval_interval = cfg["eval_interval"]
    num_eval_episodes = cfg["num_eval_episodes"]
    warmup_steps = cfg["warmup_steps"]
    num_parallel_envs = cfg["num_parallel_envs"]

    tf_mp.enable_interactive_mode()

    def make_env():
        return SandShapingEnv()
    train_py_env = parallel_py_environment.ParallelPyEnvironment([make_env] * num_parallel_envs)
    eval_py_env  = SandShapingEnv()

    vis_env = SandShapingEnv()
    max_amp = vis_env._amplitude_range[1]

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    conv_layer_params = tuple(tuple(p) for p in cfg["conv_layer_params"])
    fc_layer_params = tuple(cfg["fc_layer_params"])

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
        target_update_tau=cfg["target_update_tau"],
        target_update_period=cfg["target_update_period"],
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=cfg["reward_scale_factor"],
        train_step_counter=global_step
    )
    tf_agent.initialize()

    log_interval = 1000
    checkpoint_dir = 'ckpts'
    policy_base    = 'policies'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(policy_base, exist_ok=True)
    policy_saver = PolicySaver(tf_agent.policy)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=num_parallel_envs,
        max_length=replay_buffer_capacity
    )
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2,
        single_deterministic_pass=False
    ).prefetch(50)
    iterator = iter(dataset)

    collect_driver = DynamicStepDriver(
        train_env,
        tf_agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration
    )

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    train_checkpointer = Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=tf_agent,
        policy=tf_agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    train_checkpointer.initialize_or_restore()

    warm_batches = warmup_steps // num_parallel_envs
    for _ in range(warm_batches):
        collect_driver.run()

    vis_interval = cfg["vis_interval"]
    if vis_interval > 0:
        fig_vis, axes_vis = plt.subplots(1, 3, figsize=(15, 5))

    @function
    def train_step():
        collect_driver.run()
        experience, _ = next(iterator)
        return tf_agent.train(experience).loss

    for step in trange(1, num_iterations + 1, desc='Training'):
        train_loss = train_step()
        vis_env.step(sample_random_action(vis_env.action_spec()))
        if step % log_interval == 0:
            tqdm.write(f'step {step}: train_loss = {float(train_loss):.4f}')
        if vis_interval > 0 and step % vis_interval == 0:
            env_raw = vis_env._env_map.map
            target_raw = vis_env._target_map.map
            diff_img = vis_env._env_map.difference(vis_env._target_map)
            
            vmin, vmax = env_raw.min(), env_raw.max()
            axes_vis[0].clear()
            axes_vis[0].imshow(env_raw - np.mean(env_raw), cmap='viridis')
            axes_vis[0].set_title(f'Env @ step {step} | min:{vmin:.1f}, max:{vmax:.1f}')

            vmin, vmax = target_raw.min(), target_raw.max()
            axes_vis[1].clear()
            axes_vis[1].imshow(target_raw - np.mean(target_raw), cmap='viridis')
            axes_vis[1].set_title(f'Target | min:{vmin:.1f}, max:{vmax:.1f}')

            vmin, vmax = diff_img.min(), diff_img.max()
            axes_vis[2].clear()
            axes_vis[2].imshow(diff_img, cmap='turbo', vmin=-max_amp/2, vmax=max_amp/2)
            axes_vis[2].set_title(f'Difference | min:{vmin:.1f}, max:{vmax:.1f}, rmse = {np.sqrt(np.sum(np.square(diff_img))):.1f}')

            if plt.get_fignums():
                fig_vis.canvas.draw()
            plt.pause(0.001)
        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            tqdm.write(f'step={step}: avg_return={float(avg_return):.2f}')
            train_checkpointer.save(global_step)
            tqdm.write(f'Checkpoint saved at step {step}')

    final_path = os.path.join(policy_base, 'final')
    PolicySaver(tf_agent.policy).save(final_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to config JSON')
    args = parser.parse_args()
    train(config_path=args.config)