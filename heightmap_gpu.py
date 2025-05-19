import cupy as cp                   # GPU arrays
from cupy.random import default_rng
from functools import lru_cache


# ---------------------------------------------------------------------------
#  Perlin noise utilities (GPU)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _get_perlin_gradients(res_0, res_1, seed):
    """Cached GPU gradients via CuPy RNG."""
    rng = default_rng(seed)
    angles = 2 * cp.pi * rng.random((res_0 + 1, res_1 + 1), dtype=cp.float32)
    grads = cp.stack((cp.cos(angles), cp.sin(angles)), axis=-1)
    return grads


def _fade(t):
    """Quintic smoothing curve used by classic Perlin noise."""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def generate_perlin_noise_2d(shape, res, amplitude=1.0, seed=None):
    """
    Generate a 2‑D Perlin noise height‑field directly on the GPU.

    Parameters
    ----------
    shape : (H, W) of the output array
    res   : (rx, ry) controlling feature size
    """
    H, W = shape
    rx, ry = int(res[0]), int(res[1])

    xs = cp.linspace(0, rx, W, endpoint=False, dtype=cp.float32)
    ys = cp.linspace(ry, 0, H, endpoint=False, dtype=cp.float32)
    xv, yv = cp.meshgrid(xs, ys)

    xi = cp.floor(xv).astype(cp.int32)
    yi = cp.floor(yv).astype(cp.int32)
    xf = xv - xi
    yf = yv - yi

    u = _fade(xf)
    v = _fade(yf)

    # Move gradients for this (rx,ry,seed) to GPU once and reuse
    grads = _get_perlin_gradients(rx, ry, seed)

    def dot_grad(ix, iy, x, y):
        g = grads[ix % (rx + 1), iy % (ry + 1)]
        return g[..., 0] * x + g[..., 1] * y

    n00 = dot_grad(xi,   yi,   xf,   yf)
    n10 = dot_grad(xi+1, yi,   xf-1, yf)
    n01 = dot_grad(xi,   yi+1, xf,   yf-1)
    n11 = dot_grad(xi+1, yi+1, xf-1, yf-1)

    x1 = n00 * (1 - u) + n10 * u
    x2 = n01 * (1 - u) + n11 * u
    noise = x1 * (1 - v) + x2 * v

    mn, mx = noise.min(), noise.max()
    noise = cp.where(mx > mn, (noise - mn) / (mx - mn), cp.zeros_like(noise))
    return (noise * cp.float32(amplitude)).astype(cp.float32)


# ---------------------------------------------------------------------------
#  HeightMap class – all operations reside on GPU via CuPy
# ---------------------------------------------------------------------------
class HeightMap:
    """
    GPU‑resident height‑field with spherical press operations.
    """

    def __init__(self,
                 width,
                 height,
                 scale=(4, 4),
                 amplitude=1.0,
                 tool_radius=5,
                 seed=None,
                 bedrock_offset=0.0):
        self.width = width
        self.height = height
        self.tool_radius = tool_radius

        # --- pre‑compute spherical press geometry (GPU) ---------------------
        r = tool_radius
        coords = cp.indices((2 * r + 1, 2 * r + 1))
        dy = coords[0] - r
        dx = coords[1] - r
        self._press_mask = (dx * dx + dy * dy) <= r * r
        dist2 = dx * dx + dy * dy
        self._press_offset = cp.zeros_like(dist2, dtype=cp.float32)
        self._press_offset[self._press_mask] = (
            r - cp.sqrt(r * r - dist2[self._press_mask])
        )

        # --- terrain --------------------------------------------------------
        self.map = (
            generate_perlin_noise_2d((height, width), scale, amplitude, seed)
            + bedrock_offset
        )

        # fast running mean for centred difference
        self._size = float(width * height)
        self._sum = float(cp.sum(self.map).get())
        self._mean = self._sum / self._size

        self.bedrock = 0.0

    # ---------------------------------------------------------------------
    #  Internal helpers
    # ---------------------------------------------------------------------
    def _update_stats(self, removed):
        self._sum -= removed
        self._mean = self._sum / self._size

    def _extract_patch(self, cx, cy):
        r = self.tool_radius
        return self.map[cy - r: cy + r + 1, cx - r: cx + r + 1]

    # ---------------------------------------------------------------------
    #  Public API
    # ---------------------------------------------------------------------
    def apply_press(self, x, y, dz):
        """
        Spherical press relative depth `dz` at (x, y).
        """
        r = self.tool_radius
        cy = int(cp.clip(cp.rint(y), r, self.height - 1 - r))
        cx = int(cp.clip(cp.rint(x), r, self.width  - 1 - r))
        h_center = float(self.map[cy, cx])

        patch = self._extract_patch(cx, cy)
        intr = cp.maximum(self.bedrock, h_center - dz + self._press_offset)
        new_patch = cp.where(self._press_mask, cp.minimum(patch, intr), patch)

        removed = float(cp.sum(patch - new_patch).get())
        self.map[cy - r: cy + r + 1, cx - r: cx + r + 1] = new_patch
        self._update_stats(removed)
        return removed

    def apply_press_abs(self, x, y, z_abs, dz_rel):
        """
        Absolute‑pose press: move tip to `z_abs` then down by `dz_rel`.
        Returns (removed_volume, touched_flag)
        """
        r = self.tool_radius
        cy = int(cp.clip(cp.rint(y), r, self.height - 1 - r))
        cx = int(cp.clip(cp.rint(x), r, self.width  - 1 - r))
        h_center = float(self.map[cy, cx])

        tip_final = max(z_abs - dz_rel, self.bedrock)
        dz_effective = h_center - tip_final
        if dz_effective <= 1e-6:
            return 0.0, False
        removed = self.apply_press(x, y, dz_effective)
        return removed, True

    def difference(self, other):
        """
        Mean‑centred difference with another HeightMap *or* array.
        Always returns a CuPy array resident on the GPU.
        """
        h1 = self.map - self._mean

        if hasattr(other, "map") and hasattr(other, "_sum"):
            h2 = other.map - (other._sum / other._size)
        else:
            arr = cp.asarray(other)
            h2 = arr - cp.mean(arr)

        return h1 - h2


# ---------------------------------------------------------------------------
#  Minimal smoke‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    hm = HeightMap(400, 400, scale=(2, 2), amplitude=40, bedrock_offset=10)
    removed = hm.apply_press(200, 200, dz=5.0)
    print("Removed volume:", removed)
    print("Map min height:", float(hm.map.min().get()))