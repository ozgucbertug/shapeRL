"""Visualization helpers for sand shaping."""

from __future__ import annotations

import numpy as np
from matplotlib import colors


def visualize(fig, axes, cbars, env, step, max_amp):
    """Update matplotlib axes with environment state visualisations."""
    h = env._env_map.map
    t = env._target_map.map
    diff = env._env_map.difference(env._target_map)
    obs = env._build_observation(diff, h, t)
    hmin, hmax = h.min(), h.max()
    tmin, tmax = t.min(), t.max()
    dlim = max_amp / 2
    norm_diff = colors.TwoSlopeNorm(vcenter=0, vmin=-dlim, vmax=dlim)
    channels = [diff, h, t, obs[..., 0], obs[..., 1], obs[..., 2]]
    cmaps = ['turbo', 'viridis', 'viridis', 'turbo', 'viridis', 'viridis']
    titles = [
        f'Diff raw\nmin:{diff.min():.1f}, max:{diff.max():.1f}, rmse={np.sqrt(np.mean(diff**2)):.1f}',
        f'Env height @ step {step}\nmin:{hmin:.1f}, max:{hmax:.1f}',
        f'Target height\nmin:{tmin:.1f}, max:{tmax:.1f}',
        'Diff channel',
        'Env channel',
        'Tgt channel',
    ]
    for idx, ax in enumerate(axes.flat):
        ax.clear()
        data = channels[idx]
        cmap = cmaps[idx]
        if idx == 0:
            im = ax.imshow(data, cmap=cmap, norm=norm_diff)
        else:
            im = ax.imshow(data, cmap=cmap)
        ax.set_title(titles[idx])
        if cbars[idx] is None:
            cbars[idx] = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            cbars[idx].mappable = im
            cbars[idx].update_normal(im)
    fig.tight_layout()


__all__ = ['visualize']
