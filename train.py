"""CLI wrapper for the Shape RL training routine."""

from shape_rl.training import train, run_cli

__all__ = ["train", "run_cli"]


def main():
    run_cli()


if __name__ == "__main__":
    main()
