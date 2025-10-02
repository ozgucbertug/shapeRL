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

    num_iterations: int = 900_000
    num_envs: int = 4
    batch_size: int = 32
    collect_steps: int = 4
    checkpoint_interval: int = 0
    eval_interval: int = 5_000
    vis_interval: int = 0
    seed: Optional[int] = 42
    heuristic_warmup: bool = True
    encoder: str = "cnn"
    debug: bool = False
    env_debug: bool = True
    log_interval: int = 1_000
    initial_collect_steps: Optional[int] = None
    profile: bool = False


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
    parser.add_argument("--checkpoint_interval", type=int, default=defaults.checkpoint_interval,
                        help="Iterations between checkpoint saves (0 disables checkpoints)")
    parser.add_argument("--eval_interval", type=int, default=defaults.eval_interval,
                        help="Iterations between evaluation runs")
    parser.add_argument("--vis_interval", type=int, default=defaults.vis_interval,
                        help="Iterations between visualisations (0 disables plots)")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="Random seed (omit to use library default stochastic behaviour)")
    parser.add_argument("--encoder", type=str, default=defaults.encoder,
                        choices=["cnn", "unet", "gated", "fpn"],
                        help="Backbone encoder architecture for actor/critic")
    parser.add_argument("--log_interval", type=int, default=defaults.log_interval,
                        help="Iterations between throughput logs")
    parser.add_argument("--initial_collect_steps", type=int, default=defaults.initial_collect_steps,
                        help="Warm-up transitions to gather before training (default auto-computed)")

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
    parser.add_argument("--profile", dest="profile", action="store_true",
                        help="Enable lightweight profiling")
    parser.add_argument("--no_profile", dest="profile", action="store_false",
                        help="Disable profiling")

    parser.set_defaults(
        heuristic_warmup=defaults.heuristic_warmup,
        debug=defaults.debug,
        env_debug=defaults.env_debug,
        profile=defaults.profile,
    )
    return parser


def _config_from_args(argv: Optional[Iterable[str]], defaults: TrainingConfig) -> TrainingConfig:
    args = _parser(defaults).parse_args(argv)
    # argparse produces a namespace convertible to the dataclass fields.
    return replace(defaults, **vars(args))


def run(config: Optional[TrainingConfig] = None) -> None:
    cfg = config or CONFIG
    _train(
        vis_interval=cfg.vis_interval,
        eval_interval=cfg.eval_interval,
        num_parallel_envs=cfg.num_envs,
        checkpoint_interval=cfg.checkpoint_interval,
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
        profile=cfg.profile,
    )


def run_cli(argv: Optional[Iterable[str]] = None) -> None:
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
