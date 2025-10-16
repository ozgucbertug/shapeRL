"""Policy implementations for Shape RL."""

from __future__ import annotations

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step


class HeuristicPressPolicy(py_policy.PyPolicy):
    """Greedy one-step policy based on the diff channel.

    Depth is computed from the normalized diff channel directly (scale-free) instead of multiplying by amplitude.
    """

    def __init__(self, time_step_spec, action_spec,
                 width: int, height: int, tool_radius: int, amp_max: float, max_push_mult: float = 1.0):
        super().__init__(time_step_spec, action_spec)
        self._w = width
        self._h = height
        self._r = tool_radius
        self._amp_max = amp_max
        self._max_push_mult = max_push_mult
        self._max_depth = self._max_push_mult * tool_radius
        self._inv_depth = 1.0 / max(self._max_depth, 1e-6)
        self._depth_gain = 1.05  # small overshoot on normalized diff.

    def _single_action(self, diff_signed: np.ndarray) -> np.ndarray:
        # Be robust to TF EagerTensor or NumPy inputs
        diff_signed = diff_signed.numpy() if hasattr(diff_signed, "numpy") else diff_signed
        actual = np.asarray(diff_signed, dtype=np.float32)
        diff_mod = actual.copy()
        r = self._r
        diff_mod[:r, :] = -np.inf
        diff_mod[-r:, :] = -np.inf
        diff_mod[:, :r] = -np.inf
        diff_mod[:, -r:] = -np.inf
        diff_mod[diff_mod <= 0.0] = -np.inf
        flat_index = np.argmax(diff_mod)
        if not np.isfinite(diff_mod.flat[flat_index]):
            return np.array([0.5, 0.5, 0.0], dtype=np.float32)
        cy, cx = np.unravel_index(flat_index, diff_mod.shape)

        denom_x = max(1.0, self._w - 1)
        denom_y = max(1.0, self._h - 1)
        x_norm = cx / denom_x
        y_norm = cy / denom_y
        margin_x = self._r / denom_x
        margin_y = self._r / denom_y
        x_norm = float(np.clip(x_norm, margin_x, 1.0 - margin_x))
        y_norm = float(np.clip(y_norm, margin_y, 1.0 - margin_y))

        peak = actual[cy, cx]
        if not np.isfinite(peak):
            peak = 0.0
        peak_pos = max(0.0, float(peak))
        # Map directly to normalized depth: depth_norm in [0,1]
        depth_norm = float(np.clip(self._depth_gain * peak_pos, 0.0, 1.0))
        dz_norm = depth_norm  # Absolute depth would be depth_norm * self._max_depth.

        return np.array([x_norm, y_norm, dz_norm], dtype=np.float32)

    def _action(self, time_step, policy_state):
        obs = time_step.observation
        # Convert to NumPy if coming from a TF env/time_step
        if hasattr(obs, "numpy"):
            obs_np = obs.numpy()
        else:
            try:
                obs_np = np.asarray(obs)
            except Exception:
                # Fallback: leave as-is (PyEnvironment should already be NumPy)
                obs_np = obs
        if getattr(obs_np, "ndim", None) == 4:
            batch_actions = [self._single_action(obs_np[i, ..., 0]) for i in range(obs_np.shape[0])]
            act = np.stack(batch_actions, axis=0)
        else:
            act = self._single_action(obs_np[..., 0])
        return policy_step.PolicyStep(act, policy_state, ())


__all__ = ["HeuristicPressPolicy"]
