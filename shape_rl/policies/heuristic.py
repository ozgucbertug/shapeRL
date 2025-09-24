"""Heuristic policies for the sand shaping task."""

from __future__ import annotations

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step


class HeuristicPressPolicy(py_policy.PyPolicy):
    """Greedy one-step policy based on the diff channel."""

    def __init__(self, time_step_spec, action_spec,
                 width, height, tool_radius, amp_max, max_push_mult=1):
        super().__init__(time_step_spec, action_spec)
        self._w = width
        self._h = height
        self._r = tool_radius
        self._amp_max = amp_max
        self._max_push_mult = max_push_mult
        self._max_depth = self._max_push_mult * tool_radius
        self._inv_depth = 1.0 / max(self._max_depth, 1e-6)
        self._diff_scale = self._amp_max * 0.5
        self._depth_gain = 1.05

    def _single_action(self, diff_signed):
        actual = diff_signed.astype(np.float32) * self._diff_scale
        diff_mod = actual.copy()
        r = self._r
        diff_mod[:r, :] = -np.inf
        diff_mod[-r:, :] = -np.inf
        diff_mod[:, :r] = -np.inf
        diff_mod[:, -r:] = -np.inf
        diff_mod[diff_mod <= 0.0] = -np.inf
        flat_index = np.argmax(diff_mod)
        if not np.isfinite(diff_mod.flat[flat_index]):
            return np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
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
        depth = max(0.0, peak * self._depth_gain)
        depth = float(np.clip(depth, 0.0, self._max_depth))
        dz_norm = float(np.clip(depth * self._inv_depth, 0.0, 1.0))

        return np.array([x_norm, y_norm, dz_norm, 0.0], dtype=np.float32)

    def _action(self, time_step, policy_state):
        obs = time_step.observation
        if obs.ndim == 4:
            batch_actions = [self._single_action(obs[i, ..., 0])
                             for i in range(obs.shape[0])]
            act = np.stack(batch_actions, axis=0)
        else:
            act = self._single_action(obs[..., 0])
        return policy_step.PolicyStep(act, policy_state, ())


__all__ = ["HeuristicPressPolicy"]
