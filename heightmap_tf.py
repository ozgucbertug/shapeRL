"""
heightmap_tf.py   –  TensorFlow implementation of GPU-resident HeightMap
-----------------------------------------------------------------------
All math runs on whatever device TensorFlow chooses (GPU if available).
No CuPy / NumPy copies are performed inside the hot path.
XLA compilation is intentionally disabled for compatibility.
"""

from functools import lru_cache
from typing import Tuple, Union

import tensorflow as tf
import math

# -------------------------------------------------------------------------
#  Perlin-noise utilities (TensorFlow)
# -------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _get_perlin_gradients(res_x: int, res_y: int, seed: int | None):
    """Deterministic, cached gradient grid living **on device**."""
    rng = tf.random.Generator.from_seed(seed or 0)
    angles = rng.uniform(shape=(res_x + 1, res_y + 1),
                         minval=0.0,
                         maxval=2.0 * math.pi,
                         dtype=tf.float32)
    grads = tf.stack([tf.math.cos(angles), tf.math.sin(angles)], axis=-1)  # (rx+1, ry+1, 2)
    return grads                                                   # stays on whichever device called it


def _fade(t: tf.Tensor) -> tf.Tensor:
    """Quintic fade used in classic Perlin noise."""
    return 6. * t**5 - 15. * t**4 + 10. * t**3


def generate_perlin_noise_2d(shape: Tuple[int, int],
                             res: Tuple[int, int],
                             amplitude: float = 1.0,
                             seed: int | None = None) -> tf.Tensor:
    """
    Vectorised Perlin-noise generator – **pure TensorFlow**.

    Parameters
    ----------
    shape : (H, W)
    res   : (rx, ry)  number of periods along X / Y
    """
    H, W = shape
    rx, ry = int(res[0]), int(res[1])

    # Normalised lattice coordinates
    xs = tf.linspace(0., float(rx), W)           # [W]
    ys = tf.linspace(float(ry), 0., H)           # [H]
    xv, yv = tf.meshgrid(xs, ys)                 # [H, W]

    xi = tf.cast(tf.floor(xv), tf.int32)
    yi = tf.cast(tf.floor(yv), tf.int32)
    xf = xv - tf.cast(xi, tf.float32)
    yf = yv - tf.cast(yi, tf.float32)

    u = _fade(xf)
    v = _fade(yf)

    grads = _get_perlin_gradients(rx, ry, seed)

    # —— helper to dot gradient with delta —————————————
    def dot_grad(ix, iy, x, y):
        g = tf.gather_nd(grads, tf.stack([ix % (rx + 1), iy % (ry + 1)], axis=-1))
        return tf.reduce_sum(g * tf.stack([x, y], axis=-1), axis=-1)

    n00 = dot_grad(xi,     yi,     xf,     yf)
    n10 = dot_grad(xi + 1, yi,     xf - 1, yf)
    n01 = dot_grad(xi,     yi + 1, xf,     yf - 1)
    n11 = dot_grad(xi + 1, yi + 1, xf - 1, yf - 1)

    x1 = n00 * (1. - u) + n10 * u
    x2 = n01 * (1. - u) + n11 * u
    noise = x1 * (1. - v) + x2 * v

    # Normalise to [0, 1]
    mn = tf.reduce_min(noise)
    mx = tf.reduce_max(noise)
    noise = tf.where(mx > mn, (noise - mn) / (mx - mn), tf.zeros_like(noise))

    return tf.cast(noise * amplitude, tf.float32)        # [H, W]


# -------------------------------------------------------------------------
#  HeightMap –   mutable 2-D field with spherical “press” operator
# -------------------------------------------------------------------------
class HeightMap:
    """
    TensorFlow-backed height-field.  Internal state lives in `tf.Variable`
    so it can be updated in-place on the GPU.
    """

    def __init__(self,
                 width: int,
                 height: int,
                 scale: Tuple[float, float] = (4, 4),
                 amplitude: float = 1.0,
                 tool_radius: int = 5,
                 seed: int | None = None,
                 bedrock_offset: float = 0.0):
        self.width = width
        self.height = height
        self.tool_radius = tool_radius
        self.bedrock = tf.constant(0.0, tf.float32)

        # —— tool geometry (constants) ————————————————————————
        r = tool_radius
        coords = tf.stack(tf.meshgrid(tf.range(-r, r + 1), tf.range(-r, r + 1), indexing='ij'), axis=0)
        dy, dx = tf.cast(coords[0], tf.float32), tf.cast(coords[1], tf.float32)
        dist2 = dx**2 + dy**2
        self._press_mask   = tf.cast(dist2 <= r**2, tf.float32)                # [2r+1, 2r+1]
        self._press_offset = tf.where(self._press_mask > 0.,
                                      r - tf.sqrt(tf.clip_by_value(r**2 - dist2, 0., r**2)),
                                      tf.zeros_like(dist2, tf.float32))

        # —— terrain initialisation ————————————————————————————
        init_h = generate_perlin_noise_2d((height, width), scale, amplitude, seed) + bedrock_offset
        self.map = tf.Variable(init_h, trainable=False)          # mutable, on-device (GPU if available)

        # running mean (Python floats are OK)
        self._size = float(width * height)
        self._sum  = float(tf.reduce_sum(self.map).numpy())
        self._mean = self._sum / self._size

    # ---------------------------------------------------------------------
    #  Internal helpers (no Python loops inside @tf.function)
    # ---------------------------------------------------------------------
    @tf.function(experimental_compile=False)
    def _apply_press_kernel(self,
                            map_var: tf.Variable,
                            press_mask: tf.Tensor,
                            press_offset: tf.Tensor,
                            y0: tf.Tensor, x0: tf.Tensor,
                            dz: tf.Tensor) -> tf.Tensor:
        """
        Vectorised in-place spherical carve.
        Returns removed volume (scalar float32).
        """
        r = self.tool_radius
        patch = map_var[y0: y0 + 2*r + 1, x0: x0 + 2*r + 1]          # [2r+1,2r+1]
        h_center = patch[r, r]
        intr = tf.maximum(self.bedrock, h_center - dz + press_offset)

        new_patch = tf.where(press_mask > 0.,
                             tf.minimum(patch, intr),
                             patch)

        removed = tf.reduce_sum(patch - new_patch)

        rows = tf.range(y0, y0 + 2*r + 1)
        cols = tf.range(x0, x0 + 2*r + 1)
        yy, xx = tf.meshgrid(rows, cols, indexing='ij')
        indices = tf.stack([yy, xx], axis=-1)                     # (2r+1,2r+1,2)
        updated = tf.tensor_scatter_nd_update(map_var,
                                              tf.reshape(indices, [-1, 2]),
                                              tf.reshape(new_patch, [-1]))
        map_var.assign(updated)
        return removed

    # ---------------------------------------------------------------------
    #  Public API
    # ---------------------------------------------------------------------
    def apply_press(self, x: Union[float, tf.Tensor],
                    y: Union[float, tf.Tensor],
                    dz: Union[float, tf.Tensor]) -> float:
        """
        Press relative depth `dz` at (x, y).  Returns removed volume (Python float).
        """
        r  = self.tool_radius
        cy = int(tf.clip_by_value(tf.math.round(y),  r, self.height - 1 - r))
        cx = int(tf.clip_by_value(tf.math.round(x),  r, self.width  - 1 - r))

        removed = self._apply_press_kernel(self.map,
                                           self._press_mask,
                                           self._press_offset,
                                           tf.constant(cy - r),
                                           tf.constant(cx - r),
                                           tf.cast(dz, tf.float32))

        # update running mean on the Python side
        rv = float(removed.numpy())
        self._sum  -= rv
        self._mean  = self._sum / self._size
        return rv

    def apply_press_abs(self, x: float, y: float, z_abs: float, dz_rel: float):
        """
        Absolute-pose press (tip moves to z_abs, then deeper by dz_rel).
        Returns (removed_volume, touched_flag)
        """
        r  = self.tool_radius
        cy = int(tf.clip_by_value(tf.math.round(y),  r, self.height - 1 - r))
        cx = int(tf.clip_by_value(tf.math.round(x),  r, self.width  - 1 - r))
        h_center = float(self.map[cy, cx].numpy())

        tip_final = max(z_abs - dz_rel, float(self.bedrock.numpy()))
        dz_eff = h_center - tip_final
        if dz_eff <= 1e-6:
            return 0.0, False
        return self.apply_press(x, y, dz_eff), True

    def difference(self, other) -> tf.Tensor:
        """
        Mean-centred difference with another HeightMap or array.
        Always returns a **TensorFlow tensor** on the current device.
        """
        h1 = self.map - self._mean

        if hasattr(other, "map") and hasattr(other, "_sum"):
            h2 = other.map - (other._sum / other._size)
        else:
            arr = tf.convert_to_tensor(other, dtype=tf.float32)
            h2  = arr - tf.reduce_mean(arr)

        return h1 - h2


# -------------------------------------------------------------------------
#  Simple visual smoke‑test: 1 random press, save PNGs
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    tf.print("Visible devices:", tf.config.list_physical_devices())

    H, W = 128, 128
    hm = HeightMap(W, H, scale=(3, 3), amplitude=20.0, bedrock_offset=5.0)
    initial_map = hm.map.numpy()  # snapshot before any presses

    # ---- Save initial height‑field -------------------------------------
    plt.figure()
    plt.imshow(hm.map.numpy(), cmap="viridis")
    plt.title("Initial height map")
    plt.colorbar()
    plt.show()
    tf.print("Displayed initial height map")

    # ---- One random press ---------------------------------------------
    x = random.uniform(0, W - 1)
    y = random.uniform(0, H - 1)
    dz = random.uniform(0.0, hm.tool_radius)
    removed = hm.apply_press(x, y, dz)
    tf.print("Random press at (x,y,dz):", x, y, dz, "; removed:", removed)

    # ---- Save post‑press height‑field ---------------------------------
    plt.figure()
    plt.imshow(hm.map.numpy(), cmap="viridis")
    plt.title("Height map after press")
    plt.colorbar()
    plt.show()
    tf.print("Displayed height map after press")

    # ---- Save difference image ----------------------------------------
    diff = hm.difference(initial_map)      # current map minus initial snapshot
    plt.figure()
    plt.imshow(diff.numpy(), cmap="turbo")
    plt.title("Height difference (after – before)")
    plt.colorbar()
    plt.show()
    tf.print("Displayed height difference")

    tf.print("Smoke‑test completed ✓")