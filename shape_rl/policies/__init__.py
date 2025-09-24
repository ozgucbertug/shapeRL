"""Policy implementations."""

__all__ = ["HeuristicPressPolicy"]


def __getattr__(name):
    if name == "HeuristicPressPolicy":
        from .heuristic import HeuristicPressPolicy
        return HeuristicPressPolicy
    raise AttributeError(f"module 'shape_rl.policies' has no attribute {name!r}")
