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

@numba.njit(cache=True, nogil=True)
def _jit_apply_press_abs(map_arr, press_offset, press_mask,
                         r, x, y, z_abs, dz_rel, bedrock=0.0):

    height, width = map_arr.shape

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
def _jit_apply_press(map_arr, press_offset, press_mask, r, x, y, dz, bedrock=0.0):
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
        """
        # ------------------------------------------------------------------
        # Pad the internal array so the spherical tool can reach the borders
        # of the *playable* region without special‑case clamping.
        # The public width/height parameters refer to the playable area only.
        # ------------------------------------------------------------------
        self.playable_width = width
        self.playable_height = height
        self.pad = tool_radius

        # Expand the internal grid by the padding on all sides.
        width = width + 2 * self.pad
        height = height + 2 * self.pad

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
        self._full_map = generate_perlin_noise_2d((height, width), scale, amplitude, seed) + bedrock_offset
        # Work buffers & stats are defined on the *playable* patch only
        self._size      = self.playable_height * self.playable_width
        self._diff_buf  = np.empty((self.playable_height, self.playable_width), dtype=np.float32)
        self._update_stats()
        self.scale = scale
        self.amplitude = amplitude
        self.bedrock_offset = bedrock_offset
        self.bedrock = 0.0
        
    # ----------------------- internal helpers -------------------------
    def _update_stats(self):
        """Recompute running sum & mean over the playable region only."""
        playable = self._full_map[self.pad:-self.pad, self.pad:-self.pad]
        self._sum  = float(np.sum(playable))
        self._mean = self._sum / self._size

    @property
    def map(self):
        """
        Read‑only view (no copy) of the playable sub‑region
        (shape: playable_height × playable_width).  This is what
        external code should use for observations and rewards.
        """
        return self._full_map[self.pad:-self.pad, self.pad:-self.pad]

    def apply_press(self, x, y, dz):
        """
        Simulate a spherical press along world Z and carve the map.
        """
        # Convert from world‑space (0…playable_dim) to internal padded coords
        x += self.pad
        y += self.pad
        # Use JIT-accelerated carve
        removed = _jit_apply_press(self._full_map, self._press_offset, self._press_mask,
                                  self.tool_radius, x, y, dz, self.bedrock)
        self._update_stats()
        return removed

    def apply_press_abs(self, x, y, z_abs, dz_rel):
        """
        Absolute‑pose press:
        1. Move tool tip to absolute height z_abs.
        2. Push further down by dz_rel (non‑negative).
        Returns (removed_volume, touched_flag)
        """
        # Convert from world‑space (0…playable_dim) to internal padded coords
        x += self.pad
        y += self.pad
        # Call fused JIT kernel (computes centre & depth internally)
        removed, touched = _jit_apply_press_abs(
            self._full_map, self._press_offset, self._press_mask,
            self.tool_radius, x, y, z_abs, dz_rel, self.bedrock
        )
        if touched:
            self._update_stats()
        return removed, touched

    def difference(self, other, out=None):
        """
        Compute mean‑centered difference between this heightmap and another, re‑using an internal buffer unless an `out` buffer is supplied.
        """
        if out is None:
            out = np.empty((self.playable_height, self.playable_width), dtype=np.float32)

        # Self term
        view_self = self.map
        np.subtract(view_self, self._mean, out=out)

        # Other term
        if isinstance(other, HeightMap):
            view_other = other.map
            mean_other = other._mean
        else:
            view_other = np.asarray(other, dtype=np.float32)
            mean_other = float(np.mean(view_other))

        np.subtract(out, view_other - mean_other, out=out)
        return out

    def playable_view(self):
        """
        Return a view (no copy) of just the playable sub‑region,
        excluding the padding added for border carving.
        """
        return self.map

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches

    # Smoke-test for padded grid vs playable patch carve behavior
    hm = HeightMap(width=64, height=64, scale=(2, 2), bedrock_offset=20, amplitude=10,
                   tool_radius=5, seed=42)

    # Capture maps before and after a corner press
    full_before = hm._full_map.copy()
    patch_before = hm.map.copy()
    print(f"Full grid shape: {full_before.shape}")
    print(f"Patch shape: {patch_before.shape}")
    hm.apply_press(10, 10, dz=5.0)
    full_after = hm._full_map
    patch_after = hm.map
    diff_patch = patch_before - patch_after

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    titles = [
        "Full grid before", "Playable patch before", "Full grid after",
        "Playable patch after", "Difference patch (before−after)", ""
    ]
    images = [
        full_before, patch_before, full_after,
        patch_after, diff_patch, None
    ]
    cmaps = ["turbo", "turbo", "turbo", "turbo", "turbo", None]

    for ax, img, title, cmap in zip(axes.flat, images, titles, cmaps):
        ax.set_title(title)
        ax.axis("off")
        if img is not None:
            im = ax.imshow(img, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if title.startswith("Full grid"):
                # Draw rectangle marking the playable patch
                rect = patches.Rectangle(
                    (hm.pad, hm.pad),
                    hm.playable_width,
                    hm.playable_height,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)

    plt.tight_layout()
    plt.show()