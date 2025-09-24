"""Top-level package for Shape RL components."""

__all__ = [
    "HeightMap",
    "SandShapingEnv",
]


def __getattr__(name):
    if name == "HeightMap":
        from .terrain import HeightMap
        return HeightMap
    if name == "SandShapingEnv":
        from .envs import SandShapingEnv
        return SandShapingEnv
    raise AttributeError(f"module 'shape_rl' has no attribute {name!r}")
