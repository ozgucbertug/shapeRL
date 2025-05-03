import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from scipy.ndimage import rotate

def fade(t):
    """
    Fade function for Perlin noise.
    """
    return 6*t**5 - 15*t**4 + 10*t**3

def generate_perlin_noise_2d(shape, res, amplitude=1.0):
    """
    Generate a 2D Perlin noise heightmap.
    :param shape: Tuple (height, width) of the output array.
    :param res: Number of noise periods along each axis as (res_x, res_y).
    :param amplitude: Amplitude to scale the noise.
    :return: 2D numpy array of shape `shape`, values in [0, amplitude].
    """
    height, width = shape
    # Create coordinate grid in noise space
    xs = np.linspace(0, res[0], width, endpoint=False)
    ys = np.linspace(0, res[1], height, endpoint=False)
    xv, yv = np.meshgrid(xs, ys)

    # Integer grid cell coordinates
    xi = np.floor(xv).astype(int)
    yi = np.floor(yv).astype(int)
    # Local coordinates within cell
    xf = xv - xi
    yf = yv - yi

    # Compute fade curves
    u = fade(xf)
    v = fade(yf)

    # Random gradient vectors at each grid corner
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def dot_grad(ix, iy, x, y):
        g = gradients[ix % (res[0] + 1), iy % (res[1] + 1)]
        return g[..., 0] * x + g[..., 1] * y

    # Dot products at the four corners
    n00 = dot_grad(xi,   yi,   xf,   yf)
    n10 = dot_grad(xi+1, yi,   xf-1, yf)
    n01 = dot_grad(xi,   yi+1, xf,   yf-1)
    n11 = dot_grad(xi+1, yi+1, xf-1, yf-1)

    # Linear interpolation
    x1 = n00 * (1 - u) + n10 * u
    x2 = n01 * (1 - u) + n11 * u
    noise = x1 * (1 - v) + x2 * v

    # Normalize result to [0, 1] and multiply by amplitude
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise * amplitude

class HeightMap:
    """
    Heightmap representation using a 2D array.
    """
    def __init__(self, width, height, scale=(4, 4), amplitude=1.0, tool_radius=5, seed=None):
        """
        Initialize a heightmap with Perlin noise.
        :param width: Number of columns.
        :param height: Number of rows.
        :param scale: Tuple (res_x, res_y) controlling noise frequency.
        :param amplitude: Amplitude scale for noise.
        :param tool_radius: Radius of the spherical end-effector tool.
        :param seed: Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        self.width = width
        self.height = height
        self.tool_radius = tool_radius
        self.scale = scale
        self.amplitude = amplitude
        self.map = generate_perlin_noise_2d((height, width), scale, amplitude)

    def apply_press(self, x, y, z_start, dz, theta=0, phi=0, d=0):
        """
        Simulate a spherical press along world Z.
        :param x: X-coordinate (column) of press center.
        :param y: Y-coordinate (row) of press center.
        :param z_start: Starting height of sphere center (normalized 0-1).
        :param dz: Depth to press (positive value moves sphere down).
        :param theta: Unused placeholder for rotation around X.
        :param phi: Unused placeholder for rotation around Y.
        :param d: Unused placeholder for sliding distance.
        """
        z_end = z_start - dz
        r = self.tool_radius
        # bounding box
        y_min = max(0, int(np.floor(y - r)))
        y_max = min(self.height, int(np.ceil(y + r)) + 1)
        x_min = max(0, int(np.floor(x - r)))
        x_max = min(self.width, int(np.ceil(x + r)) + 1)
        # coordinate grids
        yy = np.arange(y_min, y_max)[:, None]
        xx = np.arange(x_min, x_max)[None, :]
        dx2 = (xx - x)**2
        dy2 = (yy - y)**2
        # compute intersection heights only within the tool radius
        dist2 = dx2 + dy2
        inside = dist2 <= r**2
        z_int = np.zeros_like(dist2, dtype=float)
        z_int[inside] = z_end - np.sqrt(r**2 - dist2[inside])
        # apply to submap
        sub = self.map[y_min:y_max, x_min:x_max]
        sub[inside] = np.minimum(sub[inside], z_int[inside])
        self.map[y_min:y_max, x_min:x_max] = sub

    def to_grayscale_image(self):
        """
        Convert the heightmap to a uint8 grayscale image (0-255).
        """
        # Normalize to [0,1]
        h = self.map
        h_min, h_max = h.min(), h.max()
        norm = (h - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(h)
        # Scale to [0,255]
        return (norm * 255).astype(np.uint8)

    def to_rgb_image(self, cmap='terrain'):
        """
        Convert the heightmap to an RGB image using a matplotlib colormap.
        :param cmap: Colormap name.
        :return: (H, W, 3) uint8 array.
        """
        norm = (self.map - self.map.min()) / (self.map.max() - self.map.min()) if self.map.max() > self.map.min() else np.zeros_like(self.map)
        cm = plt.get_cmap(cmap)
        rgb = cm(norm)[..., :3]  # Drop alpha channel
        return (rgb * 255).astype(np.uint8)

    def difference(self, other, zero_center=False):
        """
        Compute the per-pixel height difference between this heightmap and another.
        :param other: Another HeightMap instance or a 2D array of the same shape.
        :param zero_center: If True, subtract the mean difference so that the result is centered around zero.
        :return: A 2D numpy array of the same shape containing the difference.
        """
        # Extract the other height data
        if isinstance(other, HeightMap):
            h2 = other.map
        else:
            h2 = np.asarray(other)
        # Compute raw difference
        diff = self.map - h2
        # Optionally center around zero
        if zero_center:
            diff = diff - np.mean(diff)
        return diff

    def registered_difference(self, target, zero_center=False, angle_steps=8):
        """
        Compute the difference between this heightmap and a target, searching over rotations.
        Returns (diff, (x_offset, y_offset, angle)).
        """
        # Extract target template
        if isinstance(target, HeightMap):
            tmpl = target.map
        else:
            tmpl = np.asarray(target)
        best_corr = -np.inf
        best_offset = (0, 0)
        best_angle = 0
        best_tmpl = None
        # Search over discrete rotations
        for angle in np.linspace(0, 360, angle_steps, endpoint=False):
            tmpl_rot = rotate(tmpl, angle, reshape=False, order=1)
            corr = correlate2d(self.map, tmpl_rot, mode='valid')
            y_off, x_off = np.unravel_index(np.argmax(corr), corr.shape)
            val = corr[y_off, x_off]
            if val > best_corr:
                best_corr = val
                best_offset = (x_off, y_off)
                best_angle = angle
                best_tmpl = tmpl_rot
        # Extract matching patch
        h, w = best_tmpl.shape
        x_off, y_off = best_offset
        patch = self.map[y_off:y_off + h, x_off:x_off + w]
        # Compute difference
        diff = patch - best_tmpl
        if zero_center:
            diff = diff - np.mean(diff)
        return diff, (x_off, y_off, best_angle)

    def reset(self, seed=None):
        """
        Reset the heightmap to a new random Perlin noise state.
        :param seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        self.map = generate_perlin_noise_2d(
            (self.height, self.width), self.scale, self.amplitude)

    def get_state(self, as_rgb=False, cmap='terrain'):
        """
        Return the current heightmap as an image suitable for RL observation.
        :param as_rgb: If True, returns an RGB image; otherwise, grayscale.
        :param cmap: Colormap name for RGB output.
        :return: numpy array of shape (H, W) for grayscale or (H, W, 3) for RGB.
        """
        if as_rgb:
            return self.to_rgb_image(cmap)
        else:
            return self.to_grayscale_image()

    def compute_reward(self, target, zero_center=True, method='l2'):
        """
        Compute a scalar reward comparing this heightmap to a target patch.
        :param target: Another HeightMap or 2D array to match.
        :param zero_center: Whether to zero-center the difference.
        :param method: 'l2' (negative Euclidean norm) or 'l1' (negative sum of abs).
        :return: (reward, (x_offset, y_offset)) tuple, where reward is higher (less negative)
                 when the patch better matches the target.
        """
        diff, offset = self.registered_difference(target, zero_center=zero_center)
        if method == 'l2':
            reward = -np.linalg.norm(diff)
        elif method == 'l1':
            reward = -np.sum(np.abs(diff))
        else:
            raise ValueError(f"Unsupported method: {method}")
        return reward, offset

if __name__ == "__main__":
    # Demonstration
    hm = HeightMap(400, 400, scale=(2, 2), amplitude=10, tool_radius=20, seed=42)
    hm.apply_press(x=50, y=50, z_start=20.0, dz=10)
    # Compute and plot difference between the original and deformed maps
    other_map = HeightMap(400, 400, scale=(4, 4), amplitude=10, tool_radius=20, seed=42)
    diff = hm.difference(other_map, zero_center=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(diff, cmap='bwr', origin='lower')
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.colorbar(label='Height Difference')
    plt.title('Difference (zero-centered)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
