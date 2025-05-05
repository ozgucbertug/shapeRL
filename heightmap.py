import numpy as np
import matplotlib.pyplot as plt

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
    if seed is not None:
        np.random.seed(seed)
    height, width = shape

    res_0 = int(res[0])
    res_1 = int(res[1])

    xs = np.linspace(0, res_0, width, endpoint=False)
    ys = np.linspace(0, res_1, height, endpoint=False)
    xv, yv = np.meshgrid(xs, ys)
    xi = np.floor(xv).astype(int)
    yi = np.floor(yv).astype(int)
    xf = xv - xi
    yf = yv - yi
    u = fade(xf)
    v = fade(yf)
    angles = 2 * np.pi * np.random.rand(int(res_0) + 1, res_1 + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

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
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise * amplitude

class HeightMap:
    """
    Heightmap representation using a 2D numpy array.
    """
    def __init__(self, width, height, scale=(4, 4), amplitude=1.0, tool_radius=5, seed=None):
        """
        Initialize a heightmap with Perlin noise.
        """
        self.width = width
        self.height = height
        self.tool_radius = tool_radius
        self.scale = scale
        self.amplitude = amplitude
        self.map = generate_perlin_noise_2d((height, width), scale, amplitude, seed)

    def apply_press(self, x, y, dz, theta=0, phi=0, d=0, target=None):
        """
        Simulate a spherical press along world Z.
        """
        # Compute sphere center height so that sphere bottom is tangent to map at (x,y)
        ix = int(np.clip(round(x), 0, self.width - 1))
        iy = int(np.clip(round(y), 0, self.height - 1))
        h_map = self.map[iy, ix]
        z_start = h_map + self.tool_radius
        z_end = z_start - dz
        r = self.tool_radius
        y_min = max(0, int(np.floor(y - r)))
        y_max = min(self.height, int(np.ceil(y + r)) + 1)
        x_min = max(0, int(np.floor(x - r)))
        x_max = min(self.width, int(np.ceil(x + r)) + 1)
        # --- begin local reward setup ---
        if target is not None:
            # copy region before pressing
            sub_before = self.map[y_min:y_max, x_min:x_max].copy()
            # extract matching target region
            if hasattr(target, 'map'):
                target_arr = target.map
            else:
                target_arr = np.asarray(target)
            target_sub = target_arr[y_min:y_max, x_min:x_max]
            # mean-center before and target regions
            sub_before_centered = sub_before - np.mean(sub_before)
            target_sub_centered = target_sub - np.mean(target_sub)
            # compute squared error before
            err_before = np.sum((sub_before_centered - target_sub_centered) ** 2)
        # --- end local reward setup ---
        yy = np.arange(y_min, y_max)[:, None]
        xx = np.arange(x_min, x_max)[None, :]
        dist2 = (xx - x)**2 + (yy - y)**2
        inside = dist2 <= r**2
        z_int = np.zeros_like(dist2, dtype=float)
        z_int[inside] = z_end - np.sqrt(r**2 - dist2[inside])
        sub = self.map[y_min:y_max, x_min:x_max]
        sub[inside] = np.minimum(sub[inside], z_int[inside])
        self.map[y_min:y_max, x_min:x_max] = sub
        # --- begin local reward computation ---
        if target is not None:
            sub_after = self.map[y_min:y_max, x_min:x_max]
            sub_after_centered = sub_after - np.mean(sub_after)
            err_after = np.sum((sub_after_centered - target_sub_centered) ** 2)
            # reward is reduction in local error (positive if improvement)
            return err_before - err_after
        # --- end local reward computation ---

    def to_grayscale_image(self):
        """
        Convert the heightmap to a uint8 grayscale image.
        """
        h = self.map
        h_min, h_max = h.min(), h.max()
        norm = (h - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(h)
        return (norm * 255).astype(np.uint8)

    def to_rgb_image(self, cmap='terrain'):
        """
        Convert the heightmap to an RGB image using a matplotlib colormap.
        """
        norm = (self.map - self.map.min()) / (self.map.max() - self.map.min()) if self.map.max() > self.map.min() else np.zeros_like(self.map)
        cm = plt.get_cmap(cmap)
        rgb = cm(norm)[..., :3]
        return (rgb * 255).astype(np.uint8)

    def reset(self, seed=None):
        """
        Reset the heightmap to a new random Perlin noise state.
        """
        self.map = generate_perlin_noise_2d((self.height, self.width), self.scale, self.amplitude, seed)

    def get_state(self, as_rgb=False, cmap='terrain'):
        """
        Return the current heightmap for RL observation.
        """
        return self.to_rgb_image(cmap) if as_rgb else self.to_grayscale_image()

    def compute_reward(self, target):
        # Mean-center both maps
        diff = self.difference(target)
        # diff -= np.mean(diff)

        return diff, -np.sqrt(np.sum(np.square(diff)))


    def difference(self, other):
        """
        Compute mean-centered difference between this heightmap and another.
        :param other: Another HeightMap or a 2D array.
        :return: 2D numpy array of differences.
        """
        h1 = self.map - np.mean(self.map)
        if hasattr(other, 'map'):
            h2 = other.map
        else:
            h2 = np.asarray(other)
        h2 = h2 - np.mean(h2)

        diff = h1 - h2

        return diff
