"""Policy implementations for Shape RL."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d
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


class HeuristicFootprintPressPolicy(HeuristicPressPolicy):
    """Heuristic that selects presses using a convolutional tool footprint."""

    def __init__(
        self,
        time_step_spec,
        action_spec,
        width: int,
        height: int,
        tool_radius: int,
        amp_max: float,
        max_push_mult: float = 1.0,
        deficit_beta: float = 1.0,
    ):
        super().__init__(time_step_spec, action_spec, width, height, tool_radius, amp_max, max_push_mult)
        self._deficit_beta = float(deficit_beta)
        self._kernel = self._build_kernel()
        self._kernel_sum = float(np.sum(self._kernel))

    def _build_kernel(self) -> np.ndarray:
        r = self._r
        coords = np.indices((2 * r + 1, 2 * r + 1))
        dy = coords[0] - r
        dx = coords[1] - r
        dist2 = dx * dx + dy * dy
        mask = dist2 <= r * r

        # Approximate spherical-cap contribution for a max-depth press.
        offset = np.zeros_like(dist2, dtype=np.float32)
        offset[mask] = r - np.sqrt(r * r - dist2[mask])
        max_depth = max(self._max_depth, 1e-6)
        kernel = np.maximum(0.0, 1.0 - (offset / max_depth))
        kernel *= mask.astype(np.float32)
        return kernel.astype(np.float32)

    def _single_action(self, diff_signed: np.ndarray) -> np.ndarray:
        diff_signed = diff_signed.numpy() if hasattr(diff_signed, "numpy") else diff_signed
        diff = np.asarray(diff_signed, dtype=np.float32)

        pos = np.maximum(diff, 0.0)
        neg = np.maximum(-diff, 0.0)

        pos_conv = convolve2d(pos, self._kernel, mode="same", boundary="fill", fillvalue=0.0)
        if self._deficit_beta > 0.0:
            neg_conv = convolve2d(neg, self._kernel, mode="same", boundary="fill", fillvalue=0.0)
            score = pos_conv - self._deficit_beta * neg_conv
        else:
            score = pos_conv

        r = self._r
        score[:r, :] = -np.inf
        score[-r:, :] = -np.inf
        score[:, :r] = -np.inf
        score[:, -r:] = -np.inf

        flat_index = np.argmax(score)
        if not np.isfinite(score.flat[flat_index]):
            return np.array([0.5, 0.5, 0.0], dtype=np.float32)
        cy, cx = np.unravel_index(flat_index, score.shape)

        denom_x = max(1.0, self._w - 1)
        denom_y = max(1.0, self._h - 1)
        x_norm = cx / denom_x
        y_norm = cy / denom_y
        margin_x = self._r / denom_x
        margin_y = self._r / denom_y
        x_norm = float(np.clip(x_norm, margin_x, 1.0 - margin_x))
        y_norm = float(np.clip(y_norm, margin_y, 1.0 - margin_y))

        mean_surplus = pos_conv[cy, cx] / max(self._kernel_sum, 1e-6)
        dz_norm = float(np.clip(self._depth_gain * mean_surplus, 0.0, 1.0))

        return np.array([x_norm, y_norm, dz_norm], dtype=np.float32)


class HeuristicLookaheadPressPolicy(HeuristicPressPolicy):
    """One-step lookahead heuristic with approximate press simulation."""

    def __init__(
        self,
        time_step_spec,
        action_spec,
        width: int,
        height: int,
        tool_radius: int,
        amp_max: float,
        max_push_mult: float = 1.0,
        num_candidates: int = 128,
        depth_fracs: tuple[float, ...] = (0.35, 0.7, 1.0),
        alpha_over: float = 0.5,
    ):
        super().__init__(time_step_spec, action_spec, width, height, tool_radius, amp_max, max_push_mult)
        self._num_candidates = max(int(num_candidates), 1)
        clean_fracs = [float(f) for f in depth_fracs if f > 0.0]
        self._depth_fracs = tuple(clean_fracs) if clean_fracs else (1.0,)
        self._alpha_over = float(alpha_over)
        self._mask, self._offset_norm, self._kernel = self._build_kernel_parts()

    def _build_kernel_parts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self._r
        coords = np.indices((2 * r + 1, 2 * r + 1))
        dy = coords[0] - r
        dx = coords[1] - r
        dist2 = dx * dx + dy * dy
        mask = dist2 <= r * r
        mask_f = mask.astype(np.float32)

        offset = np.zeros_like(dist2, dtype=np.float32)
        offset[mask] = r - np.sqrt(r * r - dist2[mask])
        max_depth = max(self._max_depth, 1e-6)
        offset_norm = offset / max_depth

        kernel = np.maximum(0.0, 1.0 - offset_norm) * mask_f
        return mask_f, offset_norm.astype(np.float32), kernel.astype(np.float32)

    def _single_action(self, diff_signed: np.ndarray) -> np.ndarray:
        diff_signed = diff_signed.numpy() if hasattr(diff_signed, "numpy") else diff_signed
        diff = np.asarray(diff_signed, dtype=np.float32)

        pos = np.maximum(diff, 0.0)
        score = convolve2d(pos, self._kernel, mode="same", boundary="fill", fillvalue=0.0)

        r = self._r
        score[:r, :] = -np.inf
        score[-r:, :] = -np.inf
        score[:, :r] = -np.inf
        score[:, -r:] = -np.inf

        best_score = np.nanmax(score)
        if not np.isfinite(best_score) or best_score <= 0.0:
            return np.array([0.5, 0.5, 0.0], dtype=np.float32)

        flat = score.ravel()
        valid_indices = np.flatnonzero(flat > 0.0)
        if valid_indices.size == 0:
            return np.array([0.5, 0.5, 0.0], dtype=np.float32)

        k = min(self._num_candidates, int(valid_indices.size))
        if k < valid_indices.size:
            top_idx = np.argpartition(-flat[valid_indices], k - 1)[:k]
            cand_flat = valid_indices[top_idx]
        else:
            cand_flat = valid_indices

        best = None
        best_improve = -np.inf
        mask = self._mask
        offset_norm = self._offset_norm

        for flat_idx in cand_flat:
            cy, cx = np.unravel_index(flat_idx, score.shape)
            diff_center = float(diff[cy, cx])
            max_depth = min(1.0, max(0.0, diff_center + self._alpha_over))
            if max_depth <= 0.0:
                continue
            patch = diff[cy - r:cy + r + 1, cx - r:cx + r + 1]
            if patch.shape[0] != mask.shape[0] or patch.shape[1] != mask.shape[1]:
                continue
            surplus = np.maximum(patch, 0.0)
            if not np.any(surplus):
                continue
            for frac in self._depth_fracs:
                depth = max_depth * frac
                if depth <= 0.0:
                    continue
                removal = np.maximum(0.0, depth - offset_norm) * mask
                removal = np.minimum(removal, surplus)
                if not np.any(removal):
                    continue
                new_patch = patch - removal
                delta_sse = float(np.sum(new_patch * new_patch - patch * patch))
                improve = -delta_sse
                if improve > best_improve:
                    best_improve = improve
                    best = (cx, cy, depth)

        if best is None or best_improve <= 0.0:
            return np.array([0.5, 0.5, 0.0], dtype=np.float32)

        cx, cy, dz_norm = best
        denom_x = max(1.0, self._w - 1)
        denom_y = max(1.0, self._h - 1)
        x_norm = cx / denom_x
        y_norm = cy / denom_y
        margin_x = self._r / denom_x
        margin_y = self._r / denom_y
        x_norm = float(np.clip(x_norm, margin_x, 1.0 - margin_x))
        y_norm = float(np.clip(y_norm, margin_y, 1.0 - margin_y))
        dz_norm = float(np.clip(dz_norm, 0.0, 1.0))

        return np.array([x_norm, y_norm, dz_norm], dtype=np.float32)


__all__ = [
    "HeuristicPressPolicy",
    "HeuristicFootprintPressPolicy",
    "HeuristicLookaheadPressPolicy",
]
