import numpy as np
import matplotlib.pyplot as plt
plt.ion()

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
    xs = np.linspace(0, res[0], width, endpoint=False)
    ys = np.linspace(0, res[1], height, endpoint=False)
    xv, yv = np.meshgrid(xs, ys)
    xi = np.floor(xv).astype(int)
    yi = np.floor(yv).astype(int)
    xf = xv - xi
    yf = yv - yi
    u = fade(xf)
    v = fade(yf)
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def dot_grad(ix, iy, x, y):
        g = gradients[ix % (res[0] + 1), iy % (res[1] + 1)]
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

    def apply_press(self, x, y, z_start, dz, theta=0, phi=0, d=0):
        """
        Simulate a spherical press along world Z.
        """
        z_end = z_start - dz
        r = self.tool_radius
        y_min = max(0, int(np.floor(y - r)))
        y_max = min(self.height, int(np.ceil(y + r)) + 1)
        x_min = max(0, int(np.floor(x - r)))
        x_max = min(self.width, int(np.ceil(x + r)) + 1)
        yy = np.arange(y_min, y_max)[:, None]
        xx = np.arange(x_min, x_max)[None, :]
        dist2 = (xx - x)**2 + (yy - y)**2
        inside = dist2 <= r**2
        z_int = np.zeros_like(dist2, dtype=float)
        z_int[inside] = z_end - np.sqrt(r**2 - dist2[inside])
        sub = self.map[y_min:y_max, x_min:x_max]
        sub[inside] = np.minimum(sub[inside], z_int[inside])
        self.map[y_min:y_max, x_min:x_max] = sub

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

    def compute_reward(self, target_map, method='l2'):
        """
        Compute a scalar reward comparing this heightmap to a target heightmap.
        The reward is higher (less negative) when the mean-centered maps match more closely.
        """
        h1 = self.map - np.mean(self.map)
        if hasattr(target_map, 'map'):
            h2 = target_map.map
        else:
            h2 = np.asarray(target_map)
        h2 = h2 - np.mean(h2)
        diff = h1 - h2
        if method == 'l2':
            return -np.linalg.norm(diff)
        elif method == 'l1':
            return -np.sum(np.abs(diff))
        else:
            raise ValueError(f"Unsupported method: {method}")

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
        return h1 - h2

    def normalized_difference(self, other):
        """
        Compute normalized, mean-centered difference scaled by amplitude.
        :param other: Another HeightMap or a 2D array.
        :return: 2D numpy array of float32 values in [-1,1].
        """
        diff = self.difference(other)
        return (diff / self.amplitude).astype(np.float32)

    def get_difference_observation(self, other):
        """
        Get the difference observation for RL: shape (H, W, 1) float32.
        :param other: Another HeightMap or a 2D array.
        :return: 3D numpy array with a singleton channel dimension.
        """
        return self.normalized_difference(other)[..., np.newaxis]
