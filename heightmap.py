import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    """
    Make 3D axes have equal scale.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

def fade(t):
    """
    Fade function for Perlin noise.
    """
    return 6*t**5 - 15*t**4 + 10*t**3

def generate_perlin_noise_2d(shape, res):
    """
    Generate a 2D Perlin noise heightmap.
    :param shape: Tuple (height, width) of the output array.
    :param res: Number of noise periods along each axis as (res_x, res_y).
    :return: 2D numpy array of shape `shape`, values in [0, 1].
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

    # Normalize result to [0, 1]
    return (noise - noise.min()) / (noise.max() - noise.min())

class HeightMap:
    """
    Heightmap representation using a 2D array.
    """
    def __init__(self, width, height, scale=(4, 4), tool_radius=5, seed=None):
        """
        Initialize a heightmap with Perlin noise.
        :param width: Number of columns.
        :param height: Number of rows.
        :param scale: Tuple (res_x, res_y) controlling noise frequency.
        :param tool_radius: Radius of the spherical end-effector tool.
        :param seed: Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        self.width = width
        self.height = height
        self.tool_radius = tool_radius
        self.map = generate_perlin_noise_2d((height, width), scale)

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
        for i in range(self.height):
            for j in range(self.width):
                dx = j - x
                dy = i - y
                r2 = dx*dx + dy*dy
                if r2 < self.tool_radius**2:
                    # Compute new height under sphere
                    z_intersect = z_end - np.sqrt(self.tool_radius**2 - r2)
                    self.map[i, j] = min(self.map[i, j], z_intersect)

    def plot(self):
        """
        2D color map visualization of the heightmap.
        """
        plt.figure(figsize=(6, 5))
        plt.imshow(self.map, cmap='terrain', origin='lower')
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.colorbar(label='Height')
        plt.title('Heightmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def plot_3d(self):
        """
        3D surface plot of the heightmap.
        """
        X = np.arange(self.width)
        Y = np.arange(self.height)
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # ensure equal scaling on all axes
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
        ax.plot_surface(X, Y, self.map, rstride=1, cstride=1, cmap='viridis')
        set_axes_equal(ax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        plt.title('3D Heightmap')
        plt.show()

if __name__ == "__main__":
    # Demonstration
    hm = HeightMap(100, 100, scale=(4, 4), tool_radius=10, seed=42)
    hm.plot()
    hm.apply_press(x=50, y=50, z_start=5.0, dz=0.4)
    hm.plot_3d()
