"""Entry point for configuring and launching Shape RL training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Iterable, Optional

from tf_agents.system import system_multiprocessing as tf_mp

from shape_rl.training import train as _train


from dataclasses import dataclass

@dataclass
class TrainingConfig:
    seed: int | None = None
    encoder_type: str = 'spatial_softmax'
    num_updates: int = 250_000
    num_parallel_envs: int = 64
    collect_steps_per_update: int = 4
    batch_size: int = 256
    use_heuristic_warmup: bool = True
    initial_collect_steps: int | None = 2**17
    replay_capacity_total: int | None = 2**19
    debug: bool = False
    env_debug: bool = True
    log_interval: int = 1_000
    num_eval_episodes: int = 1
    eval_interval: int = 5_000
    log_eval_curves: bool = True


# Edit these defaults to change training behaviour without touching library code.
CONFIG = TrainingConfig()


# Re-export the core training loop for convenience.
train = _train


def _parser(defaults: TrainingConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shape RL training CLI")
    parser.add_argument("--num_updates", type=int, default=defaults.num_updates,
                        help="Number of training updates")
    parser.add_argument("--num_parallel_envs", type=int, default=defaults.num_parallel_envs,
                        help="Number of parallel environments for training")
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size,
                        help="Batch size for training updates")
    parser.add_argument("--collect_steps_per_update", type=int, default=defaults.collect_steps_per_update,
                        help="Steps collected per update across all environments")
    parser.add_argument("--num_eval_episodes", type=int, default=defaults.num_eval_episodes,
                        help="Episodes per evaluation run")
    parser.add_argument("--eval_interval", type=int, default=defaults.eval_interval,
                        help="Updates between evaluation runs")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="Random seed (omit to use library default stochastic behaviour)")
    parser.add_argument("--encoder_type", type=str, default=defaults.encoder_type,
                        choices=["cnn", "spatial_softmax", "spatial_k"],
                        help="Backbone encoder architecture for actor/critic")
    parser.add_argument("--log_interval", type=int, default=defaults.log_interval,
                        help="Updates between throughput logs")
    parser.add_argument("--initial_collect_steps", type=int, default=defaults.initial_collect_steps,
                        help="Warm-up transitions to gather before training (default auto-computed)")
    parser.add_argument("--replay_capacity_total", type=int, default=defaults.replay_capacity_total,
                        help="Total replay capacity (overrides default sizing computed from batch/parallel envs)")

    parser.add_argument("--heuristic_warmup", dest="use_heuristic_warmup", action="store_true",
                        help="Use heuristic policy to warm up the replay buffer")
    parser.add_argument("--no_heuristic_warmup", dest="use_heuristic_warmup", action="store_false",
                        help="Disable heuristic warm-up; rely on random actions")
    parser.add_argument("--debug", dest="debug", action="store_true",
                        help="Enable verbose environment logging and TensorBoard scalars")
    parser.add_argument("--no_debug", dest="debug", action="store_false",
                        help="Disable debug logging")
    parser.add_argument("--env_debug", dest="env_debug", action="store_true",
                        help="Enable env debug mode (passes through to SandShapingEnv)")
    parser.add_argument("--no_env_debug", dest="env_debug", action="store_false",
                        help="Disable env debug mode for faster execution")
    parser.add_argument("--log_eval_curves", action="store_true",
                        default=defaults.log_eval_curves,
                        help="Enable per-step eval curve logging to TensorBoard (default: off)")
    parser.set_defaults(
        use_heuristic_warmup=defaults.use_heuristic_warmup,
        debug=defaults.debug,
        env_debug=defaults.env_debug,
    )
    return parser


def _config_from_args(argv: Optional[Iterable[str]], defaults: TrainingConfig) -> TrainingConfig:
    if argv is None:
        args_list = None
    else:
        args_list = list(argv)
        if args_list:
            args_list = args_list[1:]
    args = _parser(defaults).parse_args(args_list)
    # argparse produces a namespace convertible to the dataclass fields.
    return replace(defaults, **vars(args))


def run(config: Optional[TrainingConfig] = None) -> None:
    cfg = config or CONFIG
    _train(
        eval_interval=cfg.eval_interval,
        num_parallel_envs=cfg.num_parallel_envs,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        collect_steps_per_update=cfg.collect_steps_per_update,
        num_updates=cfg.num_updates,
        num_eval_episodes=cfg.num_eval_episodes,
        use_heuristic_warmup=cfg.use_heuristic_warmup,
        encoder_type=cfg.encoder_type,
        debug=cfg.debug,
        log_interval=cfg.log_interval,
        initial_collect_steps=cfg.initial_collect_steps,
        env_debug=cfg.env_debug,
        replay_capacity_total=cfg.replay_capacity_total,
        log_eval_curves=cfg.log_eval_curves,
    )


def run_cli(argv: Optional[Iterable[str]] = None) -> None:
    if not tf_mp.multiprocessing_core.initialized():
        tf_mp.enable_interactive_mode()
    run(_config_from_args(argv, CONFIG))


def main(argv: Optional[Iterable[str]] = None) -> None:
    run_cli(argv)


__all__ = [
    "TrainingConfig",
    "CONFIG",
    "train",
    "run",
    "run_cli",
    "main",
]


if __name__ == "__main__":
    main()
