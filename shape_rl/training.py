import math
import os
import time
from contextlib import suppress

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm, trange

# Additional keras imports for encoder architectures
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
from tf_agents.utils.common import function
from tf_agents.policies import random_tf_policy
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.ddpg.critic_network import CriticNetwork
import tensorflow_probability as tfp


from shape_rl.envs import SandShapingEnv
from shape_rl.metrics import compute_eval, wasserstein_distance_2d
from shape_rl.policies import HeuristicPressPolicy
from datetime import datetime

# Import network architectures
from shape_rl.networks import (
    FPNActorNetwork,
    FPNCriticNetwork,
    SpatialSoftmaxActorNetwork,
    SpatialSoftmaxCriticNetwork,
)

__all__ = ["train"]

def _make_env(seed: int | None, debug: bool = False) -> callable:
    """Factory returning a thunk that creates a SandShapingEnv."""
    def _thunk():
        return SandShapingEnv(debug=debug, seed=seed)

    return _thunk


def train(
    num_parallel_envs: int = 4,
    eval_interval: int = 1000,
    seed: int | None = None,
    batch_size: int = 16,
    collect_steps_per_iteration: int = 4,
    num_iterations: int = 200000,
    use_heuristic_warmup: bool = False,
    encoder_type: str = 'cnn',
    debug: bool = False,
    log_interval: int = 1000,
    initial_collect_steps: int | None = None,
    env_debug: bool = True,
    replay_capacity_total: int | None = None,
):
    tqdm.write("[Init] Starting training setup")
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tqdm.write(f"[Init] Seeding numpy/tf with base seed {seed}")
    else:
        tqdm.write("[Init] No RNG seed provided; using default randomness")
    # Hyperparameters
    env_divisor = max(1, num_parallel_envs)
    if replay_capacity_total is not None and replay_capacity_total <= 0:
        raise ValueError("replay_capacity_total must be positive when provided")
    if replay_capacity_total is not None:
        target_total_replay = int(replay_capacity_total)
    else:
        target_total_replay = max(4096 * 4, batch_size * 64)
    replay_buffer_capacity = max(1, math.ceil(target_total_replay / env_divisor))
    total_replay_capacity = replay_buffer_capacity * env_divisor
    learning_rate = 1e-4
    gamma = 0.99
    num_eval_episodes = 5
    tqdm.write(
        "[Init] Derived hyperparameters — "
        f"replay_capacity_per_env={replay_buffer_capacity}, "
        f"total_replay_capacity={total_replay_capacity}, "
        f"collect_steps_per_iteration={collect_steps_per_iteration}"
    )

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
    tqdm.write(
        f"[Env] Building {num_parallel_envs} parallel environment factories "
        f"(env_debug={env_debug})"
    )
    env_fns = []
    for idx in range(num_parallel_envs):
        env_seed = None if seed is None else seed + idx
        env_fns.append(_make_env(env_seed, debug=env_debug))

    tqdm.write("[Env] Spawning ParallelPyEnvironment instances")
    train_py_env = ParallelPyEnvironment(env_fns, start_serially=True)
    python_envs = getattr(train_py_env, '_envs', None)

    eval_seed = seed if seed is not None else None
    tqdm.write("[Env] Creating evaluation SandShapingEnv")
    eval_py_env = SandShapingEnv(seed=eval_seed, debug=env_debug)

    eval_base_seed = (seed + 1000) if seed is not None else None

    def eval_env_factory(s):
        env_seed = s if s is not None else None
        return SandShapingEnv(seed=env_seed, debug=False)

    tqdm.write("[Env] Wrapping Python environments with TFPyEnvironment")
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Network architecture
    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()

    encoder_label = encoder_type
    if encoder_type == 'fpn':
        actor_net = FPNActorNetwork(observation_spec, action_spec)
        critic_net = FPNCriticNetwork(observation_spec, action_spec)
    elif encoder_type in ('spatial', 'spatial_softmax'):
        actor_net = SpatialSoftmaxActorNetwork(observation_spec, action_spec)
        critic_net = SpatialSoftmaxCriticNetwork(observation_spec, action_spec)
    elif encoder_type == 'cnn':
        conv_params = ((32, 3, 2), (64, 3, 2), (128, 3, 2))
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            input_tensor_spec=observation_spec,
            output_tensor_spec=action_spec,
            conv_layer_params=conv_params,
            fc_layer_params=(256, 128),
        )
        critic_net = CriticNetwork(
            input_tensor_spec=(observation_spec, action_spec),
            observation_conv_layer_params=conv_params,
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=(256, 128),
            name='critic_network'
        )
        encoder_label = "CNN"
    else:
        raise ValueError(f"Unknown encoder_type '{encoder_type}'.")

    global_step = tf.compat.v1.train.get_or_create_global_step()
    tqdm.write(f"[Agent] Initialising SAC agent with encoder '{encoder_label}'")
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

    # ---- logging setup ----
    log_root = 'logs'
    os.makedirs(log_root, exist_ok=True)
    logdir = os.path.join(log_root, datetime.now().strftime('%Y%m%d-%H%M%S'))
    summary_writer = tf.summary.create_file_writer(logdir)
    summary_writer.set_as_default()
    tqdm.write(f"[Logging] Writing TensorBoard summaries to {logdir}")

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
    tqdm.write("[Buffer] Replay buffer dataset ready; iterator initialised")

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
    tqdm.write("[Drivers] Collection driver prepared")

    # --- Evaluate heuristic policy performance on raw PyEnvironment ---
    tqdm.write("[Heuristic Eval] Evaluating heuristic policy baseline")
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
    mae_init_vals: list[float] = []
    mae_final_vals: list[float] = []
    w2_init_vals: list[float] = []
    w2_final_vals: list[float] = []
    deltas = []
    rels = []
    for _ in range(10):
        ts_py = eval_py_env.reset()
        diff0 = eval_py_env._env_map.difference(eval_py_env._target_map)
        rmse0 = float(np.sqrt(np.mean(diff0**2)))
        mae0 = float(np.mean(np.abs(diff0)))
        w20 = wasserstein_distance_2d(eval_py_env._env_map, eval_py_env._target_map)

        ts_step = ts_py
        while not ts_step.is_last():
            action_step = heuristic_policy.action(ts_step)
            act = action_step.action
            ts_step = eval_py_env.step(act)

        diffF = eval_py_env._env_map.difference(eval_py_env._target_map)
        rmseF = float(np.sqrt(np.mean(diffF**2)))
        maeF = float(np.mean(np.abs(diffF)))
        w2F = wasserstein_distance_2d(eval_py_env._env_map, eval_py_env._target_map)

        rmse_init_vals.append(rmse0)
        rmse_final_vals.append(rmseF)
        mae_init_vals.append(mae0)
        mae_final_vals.append(maeF)
        w2_init_vals.append(w20)
        w2_final_vals.append(w2F)

        deltas.append(rmse0 - rmseF)
        rels.append((rmse0 - rmseF) / (rmse0 + 1e-6))

    rmse_init_mean = float(np.mean(rmse_init_vals))
    rmse_final_mean = float(np.mean(rmse_final_vals))
    mae_init_mean = float(np.mean(mae_init_vals))
    mae_final_mean = float(np.mean(mae_final_vals))
    w2_init_mean = float(np.mean(w2_init_vals))
    w2_final_mean = float(np.mean(w2_final_vals))
    delta_mean = float(np.mean(deltas))
    rel_mean = float(np.mean(rels))

    tqdm.write(
        f"[Heuristic Eval] RMSE {rmse_init_mean:.4f} -> {rmse_final_mean:.4f} "
        f"(Δ {delta_mean:.4f}, rel {rel_mean:.2%}); "
        f"MAE {mae_init_mean:.4f} -> {mae_final_mean:.4f}; "
        f"W2 {w2_init_mean:.4f} -> {w2_final_mean:.4f}"
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

    # --- Simplified warm-up: collect exactly `initial_collect_steps` transitions ---
    tqdm.write(f"[Warmup] Starting {'heuristic' if use_heuristic_warmup else 'random'} warm-up for {initial_collect_steps} transitions")
    warmup_start = time.perf_counter()

    if use_heuristic_warmup:
        observers = [replay_buffer.add_batch]
        if warmup_rewards is not None:
            observers.append(_warmup_observer)
        heuristic_driver = PyDriver(
            train_py_env,
            heuristic_policy,
            observers=observers,
            max_steps=collect_steps_per_iteration * num_parallel_envs,
        )
        warmup_ts = train_py_env.reset()
        while _num_frames() < initial_collect_steps:
            warmup_ts, _ = heuristic_driver.run(warmup_ts)
    else:
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
        f"replay_capacity_total: {total_replay_capacity}",
        f"target_replay: {target_total_replay}",
        f"env_debug: {env_debug}",
    ]
    config_text = '\n'.join(config_lines)
    try:
        tf.summary.text('config/setup', tf.constant(config_text), step=0)
    except Exception:
        pass
    tqdm.write(f"[Config] {config_text.replace(chr(10), '; ')}")

    tqdm.write("[Train] Starting optimisation loop")

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
            collect_step()
            experience, _ = next(iterator)
            train_info = train_step(experience)
            train_loss = train_info.loss
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
            if eval_interval > 0 and step % eval_interval == 0:
                # Detailed evaluation metrics
                metrics = compute_eval(eval_env_factory, tf_agent.policy, num_eval_episodes, base_seed=eval_base_seed)
                # Log evaluation scalars
                tf.summary.scalar('eval/rmse_delta', metrics['rmse_delta_mean'], step=step)
                tf.summary.scalar('eval/rmse_auc_norm', metrics['rmse_auc_norm_mean'], step=step)
                tf.summary.scalar('eval/rmse_slope', metrics['rmse_slope_mean'], step=step)

                tf.summary.scalar('eval/mae_delta', metrics['mae_delta_mean'], step=step)
                tf.summary.scalar('eval/mae_auc_norm', metrics['mae_auc_norm_mean'], step=step)
                tf.summary.scalar('eval/mae_slope', metrics['mae_slope_mean'], step=step)

                tf.summary.scalar('eval/w2_delta', metrics['w2_delta_mean'], step=step)
                tf.summary.scalar('eval/w2_auc_norm', metrics['w2_auc_norm_mean'], step=step)
                tf.summary.scalar('eval/w2_slope', metrics['w2_slope_mean'], step=step)

                # Console output for quick look
                tqdm.write(
                    f"[Eval @ {step}] RMSE Δ={metrics['rmse_delta_mean']:.4f} "
                    f"(init {metrics['init_rmse_mean']:.4f} → final {metrics['final_rmse_mean']:.4f}); "
                    f"MAE Δ={metrics['mae_delta_mean']:.4f} (init {metrics['init_mae_mean']:.4f} → final {metrics['final_mae_mean']:.4f}); "
                    f"W2 Δ={metrics['w2_delta_mean']:.4f} (init {metrics['w2_init_mean']:.4f} → final {metrics['w2_final_mean']:.4f}); "
                    f"Success: {metrics['success_rate']:.2%}"
                )
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
                tqdm.write(
                    f"[Train @ {step}] it/s={iters_per_sec:.1f} env/s={env_steps_per_sec:.1f} "
                    f"{metrics_str}"
                )

                last_log_time = now
                last_log_step = step
    except KeyboardInterrupt:
        tqdm.write(f"[Train] Interrupted at step {current_step}")
    else:
        tqdm.write("[Train] Completed optimisation loop")
    finally:
        tqdm.write("[Cleanup] Closing environments")
        with suppress(Exception):
            train_py_env.close()
        for env_to_close in (train_env, eval_env):
            with suppress(Exception):
                env_to_close.close()
