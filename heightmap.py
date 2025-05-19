import numpy as np
from numpy.random import default_rng
import numba
from functools import lru_cache

@lru_cache(maxsize=32)
def _get_perlin_gradients(res_0, res_1, seed):
    rng = default_rng(seed)
    angles = 2 * np.pi * rng.random((res_0 + 1, res_1 + 1))
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1)

def fade(t):
    """
    Fade function for Perlin noise.
    """
    return 6*t**5 - 15*t**4 + 10*t**3

def generate_perlin_noise_2d(shape, res, amplitude=1.0, seed=None):
    """
    Generate a 2D Perlin noise heightmap.
    :param shape: Tuple (height, width) of the output array.
    :param res: Number of noise periods along each axis as (res_x, res_y).
    :param amplitude: Amplitude to scale the noise.
    :param seed: Optional random seed for reproducibility.
    :return: 2D numpy array of shape `shape`, values in [0, amplitude].
    """
    height, width = shape

    res_0 = int(res[0])
    res_1 = int(res[1])

    xs = np.linspace(0, res_0, width, endpoint=False)
    ys = np.linspace(res_1, 0, height, endpoint=False)
    xv, yv = np.meshgrid(xs, ys)
    xi = np.floor(xv).astype(int)
    yi = np.floor(yv).astype(int)
    xf = xv - xi
    yf = yv - yi
    u = fade(xf)
    v = fade(yf)
    if seed is not None:
        gradients = _get_perlin_gradients(res_0, res_1, seed)
    else:
        gradients = _get_perlin_gradients(res_0, res_1, seed)

    def dot_grad(ix, iy, x, y):
        g = gradients[ix % (res_0 + 1), iy % (res_1 + 1)]
        return g[..., 0] * x + g[..., 1] * y

    n00 = dot_grad(xi,   yi,   xf,   yf)
    n10 = dot_grad(xi+1, yi,   xf-1, yf)
    n01 = dot_grad(xi,   yi+1, xf,   yf-1)
    n11 = dot_grad(xi+1, yi+1, xf-1, yf-1)

    x1 = n00 * (1 - u) + n10 * u
    x2 = n01 * (1 - u) + n11 * u
    noise = x1 * (1 - v) + x2 * v
    mn, mx = noise.min(), noise.max()
    denom = mx - mn
    if denom > 0:
        noise = (noise - mn) / denom
    else:
        noise = np.zeros_like(noise)
    noise = noise * np.float32(amplitude)
    return noise.astype(np.float32)


# JIT-accelerated spherical press carve for HeightMap
@numba.njit(cache=True, nogil=True)
def _jit_apply_press(map_arr, press_offset, press_mask, r, x, y, dz, bedrock=0):
    """
    JIT-compiled spherical press carve. Modifies map_arr in-place and returns volume removed.
    """
    height, width = map_arr.shape
    # Compute integer center, round(y) equivalent
    cy = int(y + 0.5)
    cx = int(x + 0.5)
    # Clamp center so full mask fits
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
    # Iterate over circular mask region
    for i in range(2*r + 1):
        yi = cy - r + i
        if yi < 0 or yi >= height:
            continue
        for j in range(2*r + 1):
            xj = cx - r + j
            if xj < 0 or xj >= width:
                continue
            if press_mask[i, j]:
                orig = map_arr[yi, xj]
                intr = h_center - dz + press_offset[i, j]
                # Clamp to bedrock so the terrain never goes below zero (or user‑set floor)
                if intr < bedrock:
                    intr = bedrock
                if intr < orig:
                    removed += orig - intr
                    map_arr[yi, xj] = intr
    return removed

class HeightMap:
    """
    Heightmap representation using a 2D numpy array.
    """
    def __init__(self, width, height, scale=(4, 4), amplitude=1.0,
                 tool_radius=5, seed=None, bedrock_offset=0.0):
        """
        Initialize a heightmap with Perlin noise.
        :param bedrock_offset: Vertical offset to shift the map (does not change clamp floor).
        """
        self.width = width
        self.height = height
        self.tool_radius = tool_radius
        # Precompute press mask and squared-distance grid for tool radius
        r = self.tool_radius
        coords = np.indices((2*r+1, 2*r+1))
        dy = coords[0] - r
        dx = coords[1] - r
        self._press_mask = dx*dx + dy*dy <= r*r
        self._press_dist2 = dx*dx + dy*dy
        # Precompute offset from z_start for a full sphere carve (r - sqrt(r^2 - dist2))
        self._press_offset = np.zeros_like(self._press_dist2, dtype=np.float32)
        self._press_offset[self._press_mask] = (
            r - np.sqrt(r*r - self._press_dist2[self._press_mask])
        )
        # Buffer for per-call intrusion heights to avoid reallocating
        self._z_int = np.empty_like(self._press_offset)
        self.scale = scale
        self.amplitude = amplitude
        self.bedrock_offset = bedrock_offset
        self.bedrock = 0.0
        
        self.map = generate_perlin_noise_2d((height, width), scale, amplitude, seed) + bedrock_offset
        # Running sum and count for fast mean-centered diff
        self._size = self.width * self.height
        self._sum = float(np.sum(self.map))
        self._mean = self._sum / self._size

    def apply_press(self, x, y, dz):
        """
        Simulate a spherical press along world Z and carve the map.
        """
        # Use JIT-accelerated carve
        removed = _jit_apply_press(self.map, self._press_offset, self._press_mask,
                                  self.tool_radius, x, y, dz, self.bedrock)
        # Update running sum for fast mean recompute
        self._sum -= removed
        self._mean = self._sum / self._size
        return removed

    def apply_press_abs(self, x, y, z_abs, dz_rel):
        """
        Absolute‑pose press:
        1. Move tool tip to absolute height z_abs.
        2. Push further down by dz_rel (non‑negative).
        Returns (removed_volume, touched_flag)
        """
        # Clamp z_abs not to exceed current map max (no effect)
        # Determine center pixel
        cy = int(np.clip(round(y), self.tool_radius, self.height - 1 - self.tool_radius))
        cx = int(np.clip(round(x), self.tool_radius, self.width  - 1 - self.tool_radius))
        h_center = float(self.map[cy, cx])

        tip_final = max(z_abs - dz_rel, self.bedrock)
        # If final tip is not below current surface, nothing happens
        dz_effective = h_center - tip_final
        if dz_effective <= 1e-6:
            return 0.0, False

        removed = _jit_apply_press(self.map, self._press_offset, self._press_mask,
                                   self.tool_radius, x, y, dz_effective, self.bedrock)
        self._sum -= removed
        self._mean = self._sum / self._size
        return removed, True

    def difference(self, other):
        """
        Compute mean-centered difference between this heightmap and another.
        :param other: Another HeightMap or a 2D array.
        :return: 2D numpy array of differences.
        """
        # Mean-center self using cached mean
        h1 = self.map - self._mean
        # Mean-center other
        if hasattr(other, 'map') and hasattr(other, '_sum'):
            h2 = other.map - (other._sum / other._size)
        else:
            arr = np.asarray(other)
            mean2 = float(np.sum(arr)) / arr.size
            h2 = arr - mean2
        diff = h1 - h2
        # diff -= diff.min()
        return diff

if __name__ == '__main__':
    test = HeightMap(width=400, height=400, scale=(2,2), amplitude=40, bedrock_offset=10)
    print(np.min(test.map))