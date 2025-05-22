import numpy as np
from numpy.random import default_rng
import numba
from functools import lru_cache

# --------------------------------------------------------------------
#  Cached Perlin coordinate grids to avoid repeated meshgrid creation
# --------------------------------------------------------------------
_PERLIN_GRID_CACHE: dict[tuple[tuple[int, int], tuple[int, int]], tuple[np.ndarray, ...]] = {}

def _get_perlin_grid(shape: tuple[int, int], res: tuple[int, int]):
    """
    Return cached (xi, yi, xf, yf, u, v) arrays for a given (shape, res)
    pair.  Arrays are float32 / int32 and live for the life of the process.
    """
    key = (shape, res)
    if key in _PERLIN_GRID_CACHE:
        return _PERLIN_GRID_CACHE[key]

    height, width = shape
    res_x, res_y  = res

    xs = np.linspace(0, res_x, width,  endpoint=False, dtype=np.float32)
    ys = np.linspace(res_y, 0, height, endpoint=False, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    xi = np.floor(xv).astype(np.int32)
    yi = np.floor(yv).astype(np.int32)
    xf = xv - xi
    yf = yv - yi
    u  = fade(xf)
    v  = fade(yf)

    _PERLIN_GRID_CACHE[key] = (xi, yi, xf, yf, u, v)
    return _PERLIN_GRID_CACHE[key]

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

    # Re‑use cached coordinate grids
    xi, yi, xf, yf, u, v = _get_perlin_grid(shape, (res_0, res_1))

    # Gradients: still depend on seed so cannot be cached here
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



# JIT‑accelerated absolute‑pose press that computes centre height and
# effective depth inside the kernel, then delegates to `_jit_apply_press`.
@numba.njit(cache=True, nogil=True)
def _jit_apply_press_abs(map_arr, press_offset, press_mask,
                         r, x, y, z_abs, dz_rel, bedrock=0.0):
    """
    Variant of the spherical press for absolute tool pose.
    Returns (removed_volume, touched_flag).
    """
    height, width = map_arr.shape
    # Integer centre coordinates (same rounding as Python round())
    cy = int(y + 0.5)
    cx = int(x + 0.5)
    # Clamp so full mask fits
    if cy < r:
        cy = r
    elif cy > height - 1 - r:
        cy = height - 1 - r
    if cx < r:
        cx = r
    elif cx > width - 1 - r:
        cx = width - 1 - r
    h_center = map_arr[cy, cx]

    # Effective penetration depth
    tip_final = z_abs - dz_rel
    if tip_final < bedrock:
        tip_final = bedrock
    dz_eff = h_center - tip_final
    if dz_eff <= 1e-6:
        return 0.0, False  # nothing touched

    removed = _jit_apply_press(map_arr, press_offset, press_mask,
                               r, x, y, dz_eff, bedrock)
    return removed, True

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
        # Pre-allocated work buffer for mean-centred differences (re-used each call)
        self._diff_buf = np.empty((height, width), dtype=np.float32)
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
        # Call fused JIT kernel (computes centre & depth internally)
        removed, touched = _jit_apply_press_abs(
            self.map, self._press_offset, self._press_mask,
            self.tool_radius, x, y, z_abs, dz_rel, self.bedrock
        )
        if touched:
            self._sum -= removed
            self._mean = self._sum / self._size
        return removed, touched

    def difference(self, other, out=None):
        """
        Compute mean‑centered difference between this heightmap and another, re‑using an internal buffer unless an `out` buffer is supplied.
        """
        if out is None:
            out = self._diff_buf

        # First term: (self.map - self._mean)
        np.subtract(self.map, self._mean, out=out)

        # Second term: (other.map - mean_other)
        if hasattr(other, "map") and hasattr(other, "_sum"):
            # Another HeightMap instance
            mean_other = other._sum / other._size
            np.subtract(out, other.map, out=out)
        else:
            arr = np.asarray(other, dtype=np.float32)
            mean_other = float(np.sum(arr)) / arr.size
            np.subtract(out, arr, out=out)

        # Add the mean of the second term (effectively: - ( - mean_other ))
        out += mean_other
        return out

if __name__ == '__main__':
    test = HeightMap(width=400, height=400, scale=(2,2), amplitude=40, bedrock_offset=10)
    print(np.min(test.map))