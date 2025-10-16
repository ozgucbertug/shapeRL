"""Terrain generation and manipulation utilities."""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng
import numba
from functools import lru_cache

_PERLIN_GRID_CACHE: dict[tuple[tuple[int, int], tuple[int, int]], tuple[np.ndarray, ...]] = {}


def _get_perlin_grid(shape: tuple[int, int], res: tuple[int, int]):
    """Return cached Perlin coordinate grids for a given shape and resolution."""
    key = (shape, res)
    if key in _PERLIN_GRID_CACHE:
        return _PERLIN_GRID_CACHE[key]

    height, width = shape
    res_x, res_y = res

    xs = np.linspace(0, res_x, width, endpoint=False, dtype=np.float32)
    ys = np.linspace(res_y, 0, height, endpoint=False, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    xi = np.floor(xv).astype(np.int32)
    yi = np.floor(yv).astype(np.int32)
    xf = xv - xi
    yf = yv - yi
    u = fade(xf)
    v = fade(yf)

    _PERLIN_GRID_CACHE[key] = (xi, yi, xf, yf, u, v)
    return _PERLIN_GRID_CACHE[key]


@lru_cache(maxsize=32)
def _get_perlin_gradients(res_0, res_1, seed):
    rng = default_rng(seed)
    angles = 2 * np.pi * rng.random((res_0 + 1, res_1 + 1))
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1)


def fade(t):
    """Perlin fade curve."""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def generate_perlin_noise_2d(shape, res, amplitude=1.0, seed=None):
    res_0 = int(res[0])
    res_1 = int(res[1])

    xi, yi, xf, yf, u, v = _get_perlin_grid(shape, (res_0, res_1))
    gradients = _get_perlin_gradients(res_0, res_1, seed)

    def dot_grad(ix, iy, x, y):
        g = gradients[ix % (res_0 + 1), iy % (res_1 + 1)]
        return g[..., 0] * x + g[..., 1] * y

    n00 = dot_grad(xi, yi, xf, yf)
    n10 = dot_grad(xi + 1, yi, xf - 1, yf)
    n01 = dot_grad(xi, yi + 1, xf, yf - 1)
    n11 = dot_grad(xi + 1, yi + 1, xf - 1, yf - 1)

    x1 = n00 * (1 - u) + n10 * u
    x2 = n01 * (1 - u) + n11 * u
    noise = x1 * (1 - v) + x2 * v

    mn, mx = noise.min(), noise.max()
    denom = mx - mn
    noise = (noise - mn) / denom if denom > 0 else np.zeros_like(noise)
    noise = noise * np.float32(amplitude)
    return noise.astype(np.float32)


@numba.njit(cache=True, nogil=True)
def _jit_apply_press_abs(map_arr, press_offset, press_mask, r, x, y, z_abs, dz_rel, bedrock=0.0):
    height, width = map_arr.shape

    cy = int(y + 0.5)
    cx = int(x + 0.5)
    if cy < r:
        cy = r
    elif cy > height - 1 - r:
        cy = height - 1 - r
    if cx < r:
        cx = r
    elif cx > width - 1 - r:
        cx = width - 1 - r
    h_center = map_arr[cy, cx]

    tip_final = z_abs - dz_rel
    if tip_final < bedrock:
        tip_final = bedrock
    dz_eff = h_center - tip_final
    if dz_eff <= 1e-6:
        return 0.0, False

    removed = _jit_apply_press(map_arr, press_offset, press_mask, r, x, y, dz_eff, bedrock)
    return removed, True


@numba.njit(cache=True, nogil=True)
def _jit_apply_press(map_arr, press_offset, press_mask, r, x, y, dz, bedrock=0.0):
    height, width = map_arr.shape
    cy = int(y + 0.5)
    cx = int(x + 0.5)
    if cy < r:
        cy = r
    elif cy > height - 1 - r:
        cy = height - 1 - r
    if cx < r:
        cx = r
    elif cx > width - 1 - r:
        cx = width - 1 - r
    h_center = map_arr[cy, cx]
    removed = 0.0
    for i in range(2 * r + 1):
        yi = cy - r + i
        if yi < 0 or yi >= height:
            continue
        for j in range(2 * r + 1):
            xj = cx - r + j
            if xj < 0 or xj >= width:
                continue
            if press_mask[i, j]:
                orig = map_arr[yi, xj]
                intr = h_center - dz + press_offset[i, j]
                if intr < bedrock:
                    intr = bedrock
                if intr < orig:
                    removed += orig - intr
                    map_arr[yi, xj] = intr
    return removed


class HeightMap:
    """Heightmap representation for the sand shaping environment."""

    def __init__(self, width, height, scale=(4, 4), amplitude=1.0,
                 tool_radius=5, seed=None, bedrock_offset=0.0):
        self.playable_width = width
        self.playable_height = height
        self.pad = tool_radius

        width = width + 2 * self.pad
        height = height + 2 * self.pad

        self.width = width
        self.height = height
        self.tool_radius = tool_radius

        r = self.tool_radius
        coords = np.indices((2 * r + 1, 2 * r + 1))
        dy = coords[0] - r
        dx = coords[1] - r
        self._press_mask = dx * dx + dy * dy <= r * r
        self._press_dist2 = dx * dx + dy * dy
        self._press_offset = np.zeros_like(self._press_dist2, dtype=np.float32)
        self._press_offset[self._press_mask] = (
            r - np.sqrt(r * r - self._press_dist2[self._press_mask])
        )
        self._full_map = generate_perlin_noise_2d((height, width), scale, amplitude, seed) + bedrock_offset
        self._size = self.playable_height * self.playable_width
        self._diff_buf = np.empty((self.playable_height, self.playable_width), dtype=np.float32)
        self._sum = 0.0
        self._mean = 0.0
        self.scale = scale
        self.amplitude = amplitude
        self.bedrock_offset = bedrock_offset
        self.bedrock = 0.0

        self._update_stats()

    def _update_stats(self):
        playable = self._full_map[self.pad:-self.pad, self.pad:-self.pad]
        self._sum = float(np.sum(playable))
        self._mean = self._sum / self._size

    @property
    def map(self):
        """Return a view of the playable patch (no copy)."""
        return self._full_map[self.pad:-self.pad, self.pad:-self.pad]

    def apply_press(self, x, y, dz):
        x += self.pad
        y += self.pad
        removed = _jit_apply_press(self._full_map, self._press_offset, self._press_mask,
                                   self.tool_radius, x, y, dz, self.bedrock)
        self._update_stats()
        return removed

    def apply_press_abs(self, x, y, z_abs, dz_rel):
        x += self.pad
        y += self.pad
        removed, touched = _jit_apply_press_abs(
            self._full_map, self._press_offset, self._press_mask,
            self.tool_radius, x, y, z_abs, dz_rel, self.bedrock
        )
        if touched:
            self._update_stats()
        return removed, touched

    def difference(self, other, out=None):
        if out is None:
            out = np.empty((self.playable_height, self.playable_width), dtype=np.float32)

        view_self = self.map
        np.subtract(view_self, self._mean, out=out)

        if isinstance(other, HeightMap):
            view_other = other.map
            mean_other = other._mean
        else:
            view_other = np.asarray(other, dtype=np.float32)
            mean_other = float(np.mean(view_other))

        np.subtract(out, view_other - mean_other, out=out)
        return out

    def playable_view(self):
        return self.map

    def smooth_patch(self, x: float, y: float, strength: float) -> float:
        if strength <= 1e-3:
            return 0.0
        strength = float(np.clip(strength, 0.0, 1.0))
        x += self.pad
        y += self.pad
        r = self.tool_radius
        cy = int(np.clip(round(y), r, self.height - r - 1))
        cx = int(np.clip(round(x), r, self.width - r - 1))
        patch = self._full_map[cy - r:cy + r + 1, cx - r:cx + r + 1]
        mask = self._press_mask
        masked_values = patch[mask]
        mean_val = float(np.mean(masked_values))
        blended = masked_values * (1.0 - strength) + mean_val * strength
        patch[mask] = np.maximum(blended, self.bedrock)
        self._update_stats()
        return float(np.sum(masked_values - patch[mask]))

    def clone(self):
        clone = HeightMap.__new__(HeightMap)
        clone.playable_width = self.playable_width
        clone.playable_height = self.playable_height
        clone.pad = self.pad
        clone.width = self.width
        clone.height = self.height
        clone.tool_radius = self.tool_radius
        clone._press_mask = self._press_mask
        clone._press_dist2 = self._press_dist2
        clone._press_offset = self._press_offset
        clone._full_map = self._full_map.copy()
        clone._size = self._size
        clone._diff_buf = np.empty_like(self._diff_buf)
        clone._sum = self._sum
        clone._mean = self._mean
        clone.scale = self.scale
        clone.amplitude = self.amplitude
        clone.bedrock_offset = self.bedrock_offset
        clone.bedrock = self.bedrock
        return clone


__all__ = ["HeightMap", "generate_perlin_noise_2d"]
