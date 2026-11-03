"""
Building blocks for 3D terrain scenes: flat-colored surfaces, a solid ground
block under the terrain, and fitting the scene's height to what's in it.
"""

from typing import Any

import numpy as np
from numpy import array, concatenate, full, full_like, nanmax, nanmin, zeros_like
from numpy.typing import NDArray
from plotly.graph_objects import Figure, Surface  # type: ignore


def solid_surface(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    z: NDArray[np.floating[Any]],
    color: str,
    name: str,
    opacity: float = 1.0,
) -> Surface:
    """A surface in one flat color, rather than colored by height."""
    return Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=zeros_like(z),
        colorscale=[[0, color], [1, color]],
        opacity=opacity,
        showscale=False,
        name=name,
        hovertemplate=f"{name}<extra></extra>",
    )


def perimeter(grid: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Walk the outer edge of a 2D grid once around, ending back at the first corner."""
    return concatenate([grid[0, :], grid[1:, -1], grid[-1, -2::-1], grid[-2::-1, 0]])


def ground_block(
    mesh: NDArray[np.floating[Any]],
    depth: float,
    color: str,
    name: str = "Ground",
) -> list[Surface]:
    """
    Close the terrain off into a solid block: opaque walls running down from its
    edges to a flat base `depth` below its lowest point, and the base itself.
    """
    x_mesh, y_mesh, surface = mesh
    base = float(nanmin(surface)) - depth

    x_edge, y_edge, top_edge = perimeter(x_mesh), perimeter(y_mesh), perimeter(surface)
    walls = solid_surface(
        x=array([x_edge, x_edge]),
        y=array([y_edge, y_edge]),
        z=array([top_edge, full(len(top_edge), base)]),
        color=color,
        name=name,
    )
    floor = solid_surface(x_mesh, y_mesh, full_like(surface, base), color, name)

    return [walls, floor]


def fit_scene_height(figure: Figure, z_scale: float) -> None:
    """
    Stretch the scene's height to fit every trace in `figure`, at `z_scale` of
    scene height per unit of z, so adding things above or below the terrain
    doesn't squash its relief.
    """
    z_low = min(float(nanmin(trace.z)) for trace in figure.data)
    z_high = max(float(nanmax(trace.z)) for trace in figure.data)
    figure.update_layout(
        scene_aspectratio={"x": 1, "y": 1, "z": z_scale * (z_high - z_low)}
    )
