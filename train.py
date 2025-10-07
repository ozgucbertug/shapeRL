"""Entry point for configuring and launching Shape RL training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Iterable, Optional

from tf_agents.system import system_multiprocessing as tf_mp

from shape_rl.training import train as _train


@dataclass
class TrainingConfig:
    """Container for training hyperparameters."""

    num_iterations: int = 200_000
    num_envs: int = 4
    batch_size: int = 8
    collect_steps: int = 2
    eval_interval: int = 5_000
    seed: Optional[int] = 42
    heuristic_warmup: bool = True
    encoder: str = "spatial_softmax"
    debug: bool = False
    env_debug: bool = True
    log_interval: int = 1_000
    initial_collect_steps: Optional[int] = 4_096
    replay_capacity_total: Optional[int] = 16_384


# Edit these defaults to change training behaviour without touching library code.
CONFIG = TrainingConfig()


# Re-export the core training loop for convenience.
train = _train


def _parser(defaults: TrainingConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shape RL training CLI")
    parser.add_argument("--num_iterations", type=int, default=defaults.num_iterations,
                        help="Number of training iterations")
    parser.add_argument("--num_envs", type=int, default=defaults.num_envs,
                        help="Number of parallel environments for training")
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size,
                        help="Batch size for training updates")
    parser.add_argument("--collect_steps", type=int, default=defaults.collect_steps,
                        help="Steps collected per iteration across all environments")
    parser.add_argument("--eval_interval", type=int, default=defaults.eval_interval,
                        help="Iterations between evaluation runs")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="Random seed (omit to use library default stochastic behaviour)")
    parser.add_argument("--encoder", type=str, default=defaults.encoder,
                        choices=["cnn", "fpn", "spatial", "spatial_softmax"],
                        help="Backbone encoder architecture for actor/critic")
    parser.add_argument("--log_interval", type=int, default=defaults.log_interval,
                        help="Iterations between throughput logs")
    parser.add_argument("--initial_collect_steps", type=int, default=defaults.initial_collect_steps,
                        help="Warm-up transitions to gather before training (default auto-computed)")
    parser.add_argument("--replay_capacity_total", type=int, default=defaults.replay_capacity_total,
                        help="Total replay capacity (overrides default sizing computed from batch/parallel envs)")

    parser.add_argument("--heuristic_warmup", dest="heuristic_warmup", action="store_true",
                        help="Use heuristic policy to warm up the replay buffer")
    parser.add_argument("--no_heuristic_warmup", dest="heuristic_warmup", action="store_false",
                        help="Disable heuristic warm-up; rely on random actions")
    parser.add_argument("--debug", dest="debug", action="store_true",
                        help="Enable verbose environment logging and TensorBoard scalars")
    parser.add_argument("--no_debug", dest="debug", action="store_false",
                        help="Disable debug logging")
    parser.add_argument("--env_debug", dest="env_debug", action="store_true",
                        help="Enable env debug mode (passes through to SandShapingEnv)")
    parser.add_argument("--no_env_debug", dest="env_debug", action="store_false",
                        help="Disable env debug mode for faster execution")
    parser.set_defaults(
        heuristic_warmup=defaults.heuristic_warmup,
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
        num_parallel_envs=cfg.num_envs,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        collect_steps_per_iteration=cfg.collect_steps,
        num_iterations=cfg.num_iterations,
        use_heuristic_warmup=cfg.heuristic_warmup,
        encoder_type=cfg.encoder,
        debug=cfg.debug,
        log_interval=cfg.log_interval,
        initial_collect_steps=cfg.initial_collect_steps,
        env_debug=cfg.env_debug,
        replay_capacity_total=cfg.replay_capacity_total,
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
    # Using tf_mp.handle_main introduces an absl.app run loop that can hang
    # when the script is launched via `conda run`. We rely on `run_cli` to
    # initialise TF-Agents multiprocessing instead, so invoke main directly.
    main()
