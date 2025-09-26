import argparse
import math
import os
import time
from contextlib import suppress

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm, trange

# Additional keras imports for encoder architectures
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers.legacy import Adam

# ───── HIGH-PERF SWITCHES ─────────────────────────────────────────────────────
# mixed_precision.set_global_policy('mixed_float16')        # FP16 everywhere that is safe
# tf.config.optimizer.set_jit(True)            # XLA just-in-time compilation
# -----------------------------------------------------------------------------

# TF-Agents / RL imports
from tf_agents.environments import tf_py_environment, ParallelPyEnvironment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.utils.common import function, Checkpointer
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.policies import random_tf_policy
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import actor_distribution_network, network
from tf_agents.agents.ddpg.critic_network import CriticNetwork
import tensorflow_probability as tfp


from shape_rl.envs import SandShapingEnv
from shape_rl.metrics import compute_eval, heightmap_pointcloud, chamfer_distance, earth_movers_distance
from shape_rl.policies import HeuristicPressPolicy
from shape_rl.visualization import visualize

import matplotlib.pyplot as plt
from tf_agents.system import system_multiprocessing as tf_mp
from datetime import datetime

# Import network architectures
from shape_rl.networks import (
    CarveActorNetwork, CarveCriticNetwork, build_unet_encoder, build_gated_encoder
)

__all__ = ["train", "main", "run_cli"]

def _make_env(seed: int | None, debug: bool = False) -> callable:
    """Factory returning a thunk that creates a SandShapingEnv."""
    def _thunk():
        return SandShapingEnv(debug=debug, seed=seed)

    return _thunk


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
    debug: bool = False,
    log_interval: int = 1000,
    initial_collect_steps: int | None = None,
    env_debug: bool = True,
    profile: bool = False
):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    # Hyperparameters
    replay_buffer_capacity = max(4096*4, batch_size * 64)
    learning_rate = 1e-4
    gamma = 0.99
    num_eval_episodes = 5

    min_collect_steps = max(1, math.ceil(batch_size / max(1, num_parallel_envs)))
    if collect_steps_per_iteration < min_collect_steps:
        prev_collect = collect_steps_per_iteration
        collect_steps_per_iteration = int(min_collect_steps)
        tqdm.write(
            "[Config] collect_steps_per_iteration increased "
            f"from {prev_collect} to {collect_steps_per_iteration} so each iteration "
            "collects at least one full training batch."
        )

    # Create seeded Python environments for training, evaluation, and visualization
    env_fns = []
    for idx in range(num_parallel_envs):
        env_seed = None if seed is None else seed + idx
        env_fns.append(_make_env(env_seed, debug=env_debug))

    train_py_env = ParallelPyEnvironment(env_fns, start_serially=True)
    python_envs = getattr(train_py_env, '_envs', None)

    eval_seed = seed if seed is not None else None
    eval_py_env = SandShapingEnv(seed=eval_seed, debug=env_debug)

    vis_seed = None
    if seed is not None:
        vis_seed = seed + num_parallel_envs
    vis_env = SandShapingEnv(seed=vis_seed, debug=env_debug)

    eval_base_seed = (seed + 1000) if seed is not None else None

    def eval_env_factory(s):
        env_seed = s if s is not None else None
        return SandShapingEnv(seed=env_seed, debug=False)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Network architecture
    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    # Choose encoder architecture
    critic_preprocessing_layers = None
    critic_preprocessing_combiner = None
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
        action_passthrough = layers.Lambda(lambda a: a, name='critic_action_passthrough')
        critic_preprocessing_layers = (critic_preproc, action_passthrough)

        def _concat_inputs(inputs):
            return tf.concat(inputs, axis=-1)

        critic_preprocessing_combiner = _concat_inputs
    elif encoder_type == 'gated':
        actor_conv_params = None
        critic_conv_params = None
        actor_preproc = build_gated_encoder(observation_spec.shape)
        critic_preproc = build_gated_encoder(observation_spec.shape)
        action_passthrough = layers.Lambda(lambda a: a, name='critic_action_passthrough')
        critic_preprocessing_layers = (critic_preproc, action_passthrough)

        def _concat_inputs(inputs):
            return tf.concat(inputs, axis=-1)

        critic_preprocessing_combiner = _concat_inputs
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
        critic_kwargs = dict(
            input_tensor_spec=(observation_spec, action_spec),
            observation_conv_layer_params=critic_conv_params,
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=(256, 128) if encoder_type == 'cnn' else (128,),
            name='critic_network'
        )
        if critic_preprocessing_layers is not None:
            critic_kwargs['preprocessing_layers'] = critic_preprocessing_layers
            critic_kwargs['preprocessing_combiner'] = critic_preprocessing_combiner

        critic_net = CriticNetwork(**critic_kwargs)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf_agent = sac_agent.SacAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=Adam(learning_rate, clipnorm=1.0),
        critic_optimizer=Adam(learning_rate, clipnorm=1.0),
        alpha_optimizer=Adam(learning_rate, clipnorm=1.0),
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
    if checkpoint_interval > 0:
        policy_saver = PolicySaver(tf_agent.policy, batch_size=batch_size)
    else:
        policy_saver = None
    # ---- logging setup ----
    log_root = 'logs'
    os.makedirs(log_root, exist_ok=True)
    logdir = os.path.join(log_root, datetime.now().strftime('%Y%m%d-%H%M%S'))
    summary_writer = tf.summary.create_file_writer(logdir)
    summary_writer.set_as_default()

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
        single_deterministic_pass=False,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    logical_gpus = tf.config.list_logical_devices('GPU')
    if logical_gpus:
        target_device = logical_gpus[0].name.lower().replace('device:', '')
        try:
            dataset = dataset.apply(tf.data.experimental.prefetch_to_device(target_device))
        except (AttributeError, ValueError):
            # Fallback silently if prefetch_to_device is unsupported (e.g. Metal backend)
            pass

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

    # --- Evaluate heuristic policy performance on raw PyEnvironment ---
    # Instantiate the heuristic policy once so it is available for both
    # evaluation and (optionally) warm‑up.
    heuristic_policy = HeuristicPressPolicy(
        time_step_spec=train_py_env.time_step_spec(),
        action_spec=action_spec,
        width=eval_py_env._width,
        height=eval_py_env._height,
        tool_radius=eval_py_env._tool_radius,
        amp_max=eval_py_env._amplitude_range[1],
    )
    rmse_init_vals: list[float] = []
    rmse_final_vals: list[float] = []
    chamfer_init_vals: list[float] = []
    chamfer_final_vals: list[float] = []
    emd_init_vals: list[float] = []
    emd_final_vals: list[float] = []
    deltas = []
    rels = []
    for _ in range(10):
        ts_py = eval_py_env.reset()
        diff0 = eval_py_env._env_map.difference(eval_py_env._target_map)
        rmse0 = float(np.sqrt(np.mean(diff0**2)))
        env_cloud_init = heightmap_pointcloud(eval_py_env._env_map)
        tgt_cloud = heightmap_pointcloud(eval_py_env._target_map)
        chamfer0 = chamfer_distance(env_cloud_init, tgt_cloud)
        emd0 = earth_movers_distance(env_cloud_init, tgt_cloud)

        ts_step = ts_py
        while not ts_step.is_last():
            action_step = heuristic_policy.action(ts_step)
            act = action_step.action
            ts_step = eval_py_env.step(act)

        diffF = eval_py_env._env_map.difference(eval_py_env._target_map)
        rmseF = float(np.sqrt(np.mean(diffF**2)))
        env_cloud_final = heightmap_pointcloud(eval_py_env._env_map)
        chamferF = chamfer_distance(env_cloud_final, tgt_cloud)
        emdF = earth_movers_distance(env_cloud_final, tgt_cloud)

        rmse_init_vals.append(rmse0)
        rmse_final_vals.append(rmseF)
        chamfer_init_vals.append(chamfer0)
        chamfer_final_vals.append(chamferF)
        emd_init_vals.append(emd0)
        emd_final_vals.append(emdF)

        deltas.append(rmse0 - rmseF)
        rels.append((rmse0 - rmseF) / (rmse0 + 1e-6))

    rmse_init_mean = float(np.mean(rmse_init_vals))
    rmse_final_mean = float(np.mean(rmse_final_vals))
    delta_mean = float(np.mean(deltas))
    rel_mean = float(np.mean(rels))
    chamfer_init_mean = float(np.mean(chamfer_init_vals))
    chamfer_final_mean = float(np.mean(chamfer_final_vals))
    emd_init_mean = float(np.mean(emd_init_vals))
    emd_final_mean = float(np.mean(emd_final_vals))

    tqdm.write(
        f"[Heuristic Eval] RMSE {rmse_init_mean:.4f} -> {rmse_final_mean:.4f} "
        f"(Δ {delta_mean:.4f}, rel {rel_mean:.2%}); "
        f"Chamfer {chamfer_init_mean:.4f} -> {chamfer_final_mean:.4f}; "
        f"EMD {emd_init_mean:.4f} -> {emd_final_mean:.4f}"
    )

    # Warm-up buffer: heuristic or random
    def _num_frames() -> int:
        value = replay_buffer.num_frames()
        if hasattr(value, 'numpy'):
            value = value.numpy()
        return int(value)

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), action_spec)
    random_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        random_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration
    )

    if initial_collect_steps is None:
        initial_collect_steps = max(batch_size * 2, collect_steps_per_iteration * num_parallel_envs * 4)

    warmup_mode = 'heuristic' if use_heuristic_warmup else 'random'
    tqdm.write(
        f"[Warmup] Starting {warmup_mode} warm-up for {initial_collect_steps} transitions"
    )
    warmup_start = time.perf_counter()

    warmup_rewards = [] if (debug and use_heuristic_warmup) else None

    def _warmup_observer(trajectory):
        if warmup_rewards is not None:
            try:
                reward = trajectory.reward
                reward = float(np.mean(reward))
                warmup_rewards.append(reward)
            except Exception:
                pass

    heur_target = int(initial_collect_steps * 0.6) if use_heuristic_warmup else 0
    if heur_target > 0:
        warmup_ts = train_py_env.reset()
        heuristic_steps = collect_steps_per_iteration * num_parallel_envs
        if python_envs:
            env0 = python_envs[0]
            max_steps_env = getattr(env0, '_max_steps', None)
            if isinstance(max_steps_env, int) and max_steps_env > 0:
                heuristic_steps = max(heuristic_steps, max_steps_env)
        observers = [replay_buffer.add_batch]
        if warmup_rewards is not None:
            observers.append(_warmup_observer)
        heuristic_driver = PyDriver(
            train_py_env,
            heuristic_policy,
            observers=observers,
            max_steps=heuristic_steps
        )
        while _num_frames() < heur_target:
            warmup_ts, _ = heuristic_driver.run(warmup_ts)
            warmup_ts = train_py_env.reset()

    time_step = train_env.reset()
    while _num_frames() < initial_collect_steps:
        time_step, _ = random_driver.run(time_step)

    warmup_elapsed = max(time.perf_counter() - warmup_start, 1e-6)
    collected = _num_frames()
    tqdm.write(
        f"[Warmup] Collected {collected} transitions in {warmup_elapsed:.1f}s "
        f"({collected / warmup_elapsed:.1f} steps/s)"
    )

    train_py_env.reset()
    train_env.reset()

    if warmup_rewards:
        mean_reward = float(np.mean(warmup_rewards))
        max_reward = float(np.max(warmup_rewards))
        min_reward = float(np.min(warmup_rewards))
        tqdm.write(
            f"[Warmup] Reward stats → mean:{mean_reward:.4f} min:{min_reward:.4f} max:{max_reward:.4f}"
        )

    profile_totals = None
    if profile:
        profile_totals = {
            'warmup': warmup_elapsed,
            'train_step': 0.0,
            'debug': 0.0,
            'visualize': 0.0,
            'eval': 0.0,
            'logging': 0.0,
        }

    if batch_size % num_parallel_envs != 0:
        tqdm.write(
            f"[Config] batch_size {batch_size} is not divisible by num_parallel_envs {num_parallel_envs}; "
            "consider aligning them for uniform sampling."
        )

    config_lines = [
        f"encoder: {encoder_type}",
        f"parallel_envs: {num_parallel_envs}",
        f"batch_size: {batch_size}",
        f"collect_steps_per_iteration: {collect_steps_per_iteration}",
        f"initial_collect_steps: {initial_collect_steps}",
        f"warmup: {'heuristic' if use_heuristic_warmup else 'random'}",
        f"replay_capacity: {replay_buffer_capacity}",
        f"env_debug: {env_debug}",
    ]
    config_text = '\n'.join(config_lines)
    try:
        tf.summary.text('config/setup', tf.constant(config_text), step=0)
    except Exception:
        pass
    tqdm.write(f"[Config] {config_text.replace(chr(10), '; ')}")

    tqdm.write("[Train] Starting optimisation loop")

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

    collect_step = function(collect_driver.run)

    @function
    def train_step(experience):
        return tf_agent.train(experience)

    last_log_time = time.perf_counter()
    last_log_step = 0

    current_step = 0
    try:
        for step in trange(1, num_iterations + 1, desc='Training', dynamic_ncols=True, leave=True):
            current_step = step
            step_start = time.perf_counter() if profile_totals is not None else None
            collect_step()
            experience, _ = next(iterator)
            train_info = train_step(experience)
            if profile_totals is not None and step_start is not None:
                profile_totals['train_step'] += time.perf_counter() - step_start
            train_loss = train_info.loss
            debug_start = time.perf_counter() if profile_totals is not None else None
            if debug and python_envs:
                # log per-step debug scalars to TensorBoard (only in debug single-env mode)
                step_removed = [env._last_removed_norm for env in python_envs if hasattr(env, '_last_removed_norm')]
                step_rel     = [env._last_rel_improve for env in python_envs if hasattr(env, '_last_rel_improve')]
                step_reward  = [env._last_reward for env in python_envs if hasattr(env, '_last_reward')]
                step_err_g   = [env._last_err_global for env in python_envs if hasattr(env, '_last_err_global')]
                step_err_l   = [env._last_err_local for env in python_envs if hasattr(env, '_last_err_local')]
                step_removed_abs = [env._last_removed for env in python_envs if hasattr(env, '_last_removed')]
                step_grad = [env._last_grad for env in python_envs if hasattr(env, '_last_grad')]
                step_lap = [env._last_lap for env in python_envs if hasattr(env, '_last_lap')]
                if step_removed:
                    tf.summary.scalar('train/removed_volume_norm',
                                      float(np.mean(step_removed)), step=step)
                if step_rel:
                    tf.summary.scalar('train/rel_improve',
                                      float(np.mean(step_rel)), step=step)
                if step_reward:
                    tf.summary.scalar('train/step_reward',
                                      float(np.mean(step_reward)), step=step)
                if step_err_g:
                    tf.summary.scalar('train/global_rmse',
                                      float(np.mean(step_err_g)), step=step)
                if step_err_l:
                    tf.summary.scalar('train/local_rmse',
                                      float(np.mean(step_err_l)), step=step)
                if step_removed_abs:
                    tf.summary.scalar('train/removed_volume_abs',
                                      float(np.mean(step_removed_abs)), step=step)
                if step_grad:
                    tf.summary.scalar('train/grad_rmse',
                                      float(np.mean(step_grad)), step=step)
                if step_lap:
                    tf.summary.scalar('train/lap_rmse',
                                      float(np.mean(step_lap)), step=step)

                reward_terms: dict[str, list[float]] = {}
                for env in python_envs:
                    terms = getattr(env, '_last_reward_terms', None)
                    if not terms:
                        continue
                    for key, value in terms.items():
                        reward_terms.setdefault(key, []).append(value)
                for key, values in reward_terms.items():
                    tf.summary.scalar(f'train/reward_terms/{key}',
                                      float(np.mean(values)), step=step)
            if profile_totals is not None and debug_start is not None:
                profile_totals['debug'] += time.perf_counter() - debug_start

            if vis_interval > 0 and step % vis_interval == 0:
                vis_start = time.perf_counter() if profile_totals is not None else None
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
                if profile_totals is not None and vis_start is not None:
                    profile_totals['visualize'] += time.perf_counter() - vis_start
            if eval_interval > 0 and step % eval_interval == 0:
                eval_start = time.perf_counter() if profile_totals is not None else None
                # Detailed evaluation metrics
                metrics = compute_eval(eval_env_factory, tf_agent.policy, num_eval_episodes, base_seed=eval_base_seed)
                # Log the requested evaluation cards to TensorBoard
                tf.summary.scalar('eval/rmse_delta', metrics['rmse_delta_mean'], step=step)
                tf.summary.scalar('eval/rmse_auc', metrics['rmse_auc_mean'], step=step)
                tf.summary.scalar('eval/rmse_slope', metrics['rmse_slope_mean'], step=step)
                tf.summary.scalar('eval/chamfer_delta', metrics['chamfer_delta_mean'], step=step)
                tf.summary.scalar('eval/chamfer_auc', metrics['chamfer_auc_mean'], step=step)
                tf.summary.scalar('eval/chamfer_slope', metrics['chamfer_slope_mean'], step=step)
                tf.summary.scalar('eval/emd_delta', metrics['emd_delta_mean'], step=step)
                tf.summary.scalar('eval/emd_auc', metrics['emd_auc_mean'], step=step)
                tf.summary.scalar('eval/emd_slope', metrics['emd_slope_mean'], step=step)
                # Console output for quick look
                tqdm.write(
                    f"[Eval @ {step}] RMSE Δ={metrics['rmse_delta_mean']:.4f} "
                    f"(init {metrics['init_rmse_mean']:.4f} → final {metrics['final_rmse_mean']:.4f}, rel {metrics['rel_improve_mean']:.2%}); "
                    f"Chamfer Δ={metrics['chamfer_delta_mean']:.4f} (init {metrics['chamfer_init_mean']:.4f} → final {metrics['chamfer_final_mean']:.4f}); "
                    f"EMD Δ={metrics['emd_delta_mean']:.4f} (init {metrics['emd_init_mean']:.4f} → final {metrics['emd_final_mean']:.4f}); "
                    f"Success: {metrics['success_rate']:.2%}"
                )
                if profile_totals is not None and eval_start is not None:
                    profile_totals['eval'] += time.perf_counter() - eval_start
            if log_interval > 0 and step % log_interval == 0:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-6)
                step_delta = step - last_log_step
                iters_per_sec = step_delta / elapsed
                env_steps = step_delta * collect_steps_per_iteration * num_parallel_envs
                env_steps_per_sec = env_steps / elapsed

                # Gather lightweight scalar metrics
                loss_components = []
                try:
                    total_loss = float(train_loss.numpy())
                    loss_components.append(f"loss={total_loss:.4f}")
                except Exception:
                    pass
                extra = getattr(train_info, 'extra', None)
                if extra is not None:
                    for attr in ('policy_loss', 'critic_loss', 'alpha_loss'):
                        if hasattr(extra, attr):
                            tensor = getattr(extra, attr)
                            try:
                                display_name = attr.replace('_loss', '')
                                loss_components.append(f"{display_name}={float(tensor.numpy()):.4f}")
                            except Exception:
                                continue
                try:
                    buffer_fill = _num_frames()
                    loss_components.append(f"replay={buffer_fill}")
                except Exception:
                    pass

                metrics_str = ' '.join(loss_components)
                log_start = time.perf_counter() if profile_totals is not None else None
                tqdm.write(
                    f"[Train @ {step}] it/s={iters_per_sec:.1f} env/s={env_steps_per_sec:.1f} "
                    f"{metrics_str}"
                )

                last_log_time = now
                last_log_step = step
                if profile_totals is not None and log_start is not None:
                    profile_totals['logging'] += time.perf_counter() - log_start
            # if checkpoint_interval > 0 and step % checkpoint_interval == 0:
            #     train_checkpointer.save(global_step)
            #     tqdm.write(f"[Checkpoint @ {step}] train_loss = {float(train_loss):.4f}")
    except KeyboardInterrupt:
        tqdm.write(f"[Train] Interrupted at step {current_step}")
    else:
        tqdm.write("[Train] Completed optimisation loop")
    finally:
        if profile_totals is not None:
            tqdm.write("[Profile] Timing summary (seconds)")
            for key, value in profile_totals.items():
                tqdm.write(f"[Profile] {key}: {value:.2f}")
            elapsed_total = sum(profile_totals.values())
            tqdm.write(f"[Profile] Total measured: {elapsed_total:.2f}s")

        with suppress(Exception):
            train_py_env.close()
        for env_to_close in (train_env, eval_env):
            with suppress(Exception):
                env_to_close.close()
        with suppress(Exception):
            vis_env.close()
        if vis_interval > 0:
            with suppress(Exception):
                plt.close(fig_vis)

    # final_path = os.path.join(policy_base, 'final')
    # PolicySaver(tf_agent.policy).save(final_path)


# Main entry point for multiprocessing
def main(_argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=100000,
                            help='Number of training iterations')
    parser.add_argument('--num_envs', type=int, default=4,
                            help='Number of parallel environments for training')
    parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size for training')
    parser.add_argument('--collect_steps', type=int, default=4,
                            help='Number of steps to collect per iteration')
    parser.add_argument('--checkpoint_interval', type=int, default=0,
                            help='Steps between checkpoint saves')
    parser.add_argument('--eval_interval', type=int, default=5000,
                            help='Steps between evaluation')
    parser.add_argument('--vis_interval', type=int, default=0,
                            help='Visualization interval')
    parser.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')
    parser.add_argument('--heuristic_warmup', action='store_true', default=True,
                            help='Use heuristic policy for warm-up instead of random actions')
    parser.add_argument('--encoder', type=str, default='cnn', choices=['cnn', 'unet', 'gated', 'fpn'],
                            help='Backbone encoder to use for actor/critic')
    parser.add_argument('--debug', action='store_true', default=False,
                            help='Enable environment debug bookkeeping and extra TensorBoard logging')
    parser.add_argument('--log_interval', type=int, default=1000,
                            help='Iterations between console throughput logs')
    parser.add_argument('--initial_collect_steps', type=int, default=None,
                            help='Warm-up transitions to gather before training updates')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Enable lightweight profiling of key training stages')

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
          debug=args.debug,
          log_interval=args.log_interval,
          initial_collect_steps=args.initial_collect_steps,
          env_debug=args.debug,
          profile=args.profile)


def run_cli():
    tf_mp.enable_interactive_mode()
    main()
