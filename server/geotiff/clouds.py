"""
Clouds for 3D terrain scenes. A cloud is a flat base outline plus a height; its
fluffy volume above the base is random noise drawn on creation, and it's drawn
as a solid: a smooth top over that volume, closed off by its flat base.
"""

from collections.abc import Sequence
from math import ceil
from typing import Any, Literal

import numpy as np
from numpy import (
    allclose,
    arctan2,
    argmax,
    asarray,
    clip,
    column_stack,
    concatenate,
    cos,
    dot,
    errstate,
    float32,
    float64,
    full,
    full_like,
    hypot,
    linspace,
    maximum,
    meshgrid,
    minimum,
    pi,
    roll,
    sin,
    sqrt,
    take_along_axis,
    unravel_index,
    where,
    zeros,
)
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from plotly.graph_objects import Surface  # type: ignore
from scipy.ndimage import (  # type: ignore
    gaussian_filter,
    gaussian_filter1d,
    map_coordinates,
    zoom,
)

from .scene import solid_surface

type CloudAmount = Literal["FEW", "SCT", "BKN", "OVC"]

# Share of the sky covered for each METAR cloud amount, from the middle of its okta range
CLOUD_COVERAGE: dict[CloudAmount, float] = {
    "FEW": 1.5 / 8,
    "SCT": 3.5 / 8,
    "BKN": 6 / 8,
    "OVC": 1.0,
}

BASE_WOBBLE = 0.3  # how far a random base's radius strays from its mean, either way
BODY_HEIGHT = 0.2  # share of a cloud's height its body fills, under the puffs
CLOUD_EDGE_DENSITY = 0.1  # density at a cloud's outer surface (its core is 1)
# Share of the way into its base over which a cloud's sides round off
SIDE_ROUNDING = 0.15
# Default detail of a cloud: its noise grid's points along x, y and z. The drawn
# surface follows it, so this is the one setting for how fine clouds look.
CLOUD_RESOLUTION = (48, 48, 24)
NOISE_GRID = 32  # most points per axis noise is made on, before stretching to fit
# Share of a cloud's opacity left at its base (it has all of it at its peaks)
CLOUD_FADE = 0.7
# How much cloud shapes are smoothed, as a share of each axis of their grid, so a
# finer grid draws the same cloud more finely rather than a sharper-edged one
SMOOTHING = 0.018

PLACEMENT_TRIES = 50  # random spots tried for each cloud before trying smaller ones
# Smallest clouds used to fill gaps, as a share of the size asked for; past that
# a layer's cover comes out short, rather than filled with lots of little clouds
MIN_CLOUD_SHRINK = 0.5


def normalize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rescale `values` to span 0..1."""
    span: NDArray[np.float64] = (values - values.min()) / (values.max() - values.min())
    return span


def fractal_noise(
    shape: tuple[int, ...],
    rng: Generator,
    scales: tuple[float, ...] = (0.2, 0.1, 0.05),
) -> NDArray[np.float64]:
    """
    Billow noise in 0..1, for puffy cloud tops. Each octave is random noise
    smoothed to a feature size (`scales`, as a share of each axis), then folded
    with abs() so it swells into rounded puffs; finer octaves add less.

    It's smooth at any size, so it's made on a grid of at most NOISE_GRID points
    per axis and stretched to `shape`, which keeps it quick on fine grids.
    """
    coarse = tuple(min(size, NOISE_GRID) for size in shape)
    noise = zeros(coarse)
    for octave, scale in enumerate(scales):
        field = gaussian_filter(
            rng.standard_normal(coarse), sigma=[size * scale for size in coarse]
        )
        noise += abs(field / field.std()) / 2**octave

    stretch = [size / small for size, small in zip(shape, coarse)]
    return normalize(zoom(noise, stretch, order=1))


def signed_distance(
    outline: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Distance from each (x, y) to the polygon `outline` (N x 2 points, in order
    around it): positive inside the polygon, negative outside.
    """
    start, end = outline, roll(outline, -1, axis=0)
    edge = end - start
    dx, dy = x[..., None] - start[:, 0], y[..., None] - start[:, 1]

    # Nearest point on each edge, as a share of the way along it
    along = clip((dx * edge[:, 0] + dy * edge[:, 1]) / (edge**2).sum(axis=1), 0, 1)
    distance = hypot(dx - along * edge[:, 0], dy - along * edge[:, 1]).min(axis=-1)

    # Even-odd rule: inside if a ray heading +x crosses the outline an odd number of times
    straddles = (start[:, 1] > y[..., None]) != (end[:, 1] > y[..., None])
    with errstate(divide="ignore", invalid="ignore"):
        crossing_x = start[:, 0] + dy * edge[:, 0] / edge[:, 1]
    inside = (straddles & (x[..., None] < crossing_x)).sum(axis=-1) % 2 == 1

    return where(inside, distance, -distance)


def is_star_shaped(outline: NDArray[np.float64]) -> bool:
    """
    Whether the polygon `outline` (N x 2 points) runs once around its centre
    (the mean of its points) always turning the same way, so a straight line
    from the centre meets each part of the outline only once.
    """
    offset = outline - outline.mean(axis=0)
    angles = arctan2(offset[:, 1], offset[:, 0])
    turns = (roll(angles, -1) - angles + pi) % (2 * pi) - pi
    return bool((turns > 0).all() or (turns < 0).all()) and allclose(
        abs(turns.sum()), 2 * pi
    )


def outline_points(outline: NDArray[np.float64], per_edge: int) -> NDArray[np.float64]:
    """
    Points evenly along each edge of the polygon `outline` (N x 2), keeping its
    corners, as a closed loop that ends back where it started.
    """
    start, end = outline, roll(outline, -1, axis=0)
    along = linspace(0, 1, per_edge, endpoint=False)[None, :, None]
    points = (start[:, None, :] + along * (end - start)[:, None, :]).reshape(-1, 2)
    return concatenate([points, points[:1]])


def polygon_area(outline: NDArray[np.float64]) -> float:
    """Area inside the polygon `outline` (N x 2 or N x 3 points; z is ignored)."""
    x, y = outline[:, 0], outline[:, 1]
    return float(abs(dot(x, roll(y, -1)) - dot(y, roll(x, -1))) / 2)


def random_base(
    center: tuple[float, float],
    radius: float,
    altitude: float,
    rng: Generator,
    points: int = 16,
) -> NDArray[np.float64]:
    """
    An irregular, roughly round cloud base: `points` (x, y, z) points around
    `center` at `altitude`, their distance from it wobbling around `radius`.
    """
    angles = linspace(0, 2 * pi, points, endpoint=False)
    # Smooth the wobble around the loop, wrapping at the ends, so the outline lumps rather than zigzags
    wobble = gaussian_filter1d(rng.standard_normal(points), sigma=1.5, mode="wrap")
    radii = radius * (1 + BASE_WOBBLE * wobble / abs(wobble).max())

    return column_stack(
        [
            center[0] + radii * cos(angles),
            center[1] + radii * sin(angles),
            full(points, altitude),
        ]
    )


class Cloud:
    """
    A cloud with a flat base, outlined by `base` (N x 3 points that all share
    one z), rising up to `height` above it. Its fluffy volume is drawn at random
    on a `resolution` grid when the cloud is created: `puffs` rounded lumps piled
    on a flat-bottomed body, roughened with noise. Pass a `seed` to get the same
    cloud again.

    `opacity` (0..1] is how solid the cloud is at its peaks; lower down it fades
    to CLOUD_FADE of that at its base.
    """

    __slots__ = ("base", "height", "opacity", "x", "y", "z", "density")

    def __init__(
        self,
        base: NDArray[np.float64],
        height: float,
        seed: int | None = None,
        resolution: tuple[int, int, int] = CLOUD_RESOLUTION,
        puffs: int = 24,
        opacity: float = 1.0,
    ) -> None:
        base = asarray(base, dtype=float64)
        if base.ndim != 2 or base.shape[1] != 3 or len(base) < 3:
            raise ValueError("A cloud's base needs at least 3 (x, y, z) points.")
        if not allclose(base[:, 2], base[0, 2]):
            raise ValueError("A cloud's base must be flat: all its points need one z.")
        if not is_star_shaped(base[:, :2]):
            raise ValueError(
                "A cloud's base must be star-shaped: its points need to run once "
                "around its centre, all turning the same way."
            )
        if height <= 0:
            raise ValueError("A cloud's height must be positive.")
        if not 0 < opacity <= 1:
            raise ValueError("A cloud's opacity must be above 0, and at most 1.")

        self.base = base
        self.height = float(height)
        self.opacity = float(opacity)
        self.x, self.y, self.z, self.density = self._fill(
            resolution, puffs, default_rng(seed)
        )

    def __repr__(self) -> str:
        return (
            f"Cloud(base=<{len(self.base)} points at z={self.altitude:.3g}>, "
            f"height={self.height:.3g}, opacity={self.opacity:.3g})"
        )

    def fade(self, rise: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        The cloud's opacity at `rise` (0 at its base, 1 at its full height): from
        CLOUD_FADE of its opacity at the base, quickly up to all of it at its peaks.
        """
        return self.opacity * (CLOUD_FADE + (1 - CLOUD_FADE) * sqrt(clip(rise, 0, 1)))

    @property
    def altitude(self) -> float:
        """Height of the cloud's flat base."""
        return float(self.base[0, 2])

    def _fill(
        self, resolution: tuple[int, int, int], puffs: int, rng: Generator
    ) -> tuple[NDArray[np.float64], ...]:
        """Grid over the cloud's bounds, and its density (0 = clear air, 1 = core) on it."""
        nx, ny, nz = resolution
        outline = self.base[:, :2]
        (x_min, y_min), (x_max, y_max) = outline.min(axis=0), outline.max(axis=0)

        # Pad the grid sideways, leaving room around the base for the top to come down to it
        pad_x, pad_y = 0.08 * (x_max - x_min), 0.08 * (y_max - y_min)
        x, y, z = meshgrid(
            linspace(x_min - pad_x, x_max + pad_x, nx),
            linspace(y_min - pad_y, y_max + pad_y, ny),
            linspace(self.altitude, self.altitude + self.height, nz),
            indexing="ij",
        )
        # Grid positions along each axis in 0..1, so puffs scale with the cloud whatever its units
        axes = [linspace(0, 1, n) for n in resolution]

        # How far each column is into the base: 1 at its middle, 0 on the outline, negative outside
        reach = signed_distance(outline, x[:, :, 0], y[:, :, 0])
        reach /= reach.max()

        rise = axes[2][None, None, :]
        body = clip(reach[:, :, None] / 0.15, 0, 1) * clip(1 - rise / BODY_HEIGHT, 0, 1)
        lumps = self._puffs(axes, reach, puffs, rng)
        texture = 0.75 + 0.5 * fractal_noise(x.shape, rng, scales=(0.12, 0.06))
        # Smoothing rounds off the creases between puffs, so the surface drawn is smooth
        density = gaussian_filter(
            clip(maximum(body, lumps) * texture, 0, 1),
            sigma=[SMOOTHING * size for size in resolution],
        )

        # Keep the cloud within its base outline, all the way up
        density *= clip(reach[:, :, None] / 0.1, 0, 1)

        return x, y, z, density

    @staticmethod
    def _puffs(
        axes: list[NDArray[np.floating[Any]]],
        reach: NDArray[np.float64],
        count: int,
        rng: Generator,
    ) -> NDArray[np.float64]:
        """
        `count` overlapping round puffs over the base, each 1 at its centre fading
        to 0 at its edge. They land more often towards the middle of the base, and
        stand taller there, so the cloud heaps up into a cauliflower top.
        """
        weights = sqrt(clip(reach, 0, None)).ravel()
        cells = rng.choice(weights.size, size=count, p=weights / weights.sum())

        density = zeros([len(axis) for axis in axes])
        for cell in cells:
            i, j = unravel_index(cell, reach.shape)
            depth = reach[i, j]
            radius = rng.uniform(0.12, 0.26) * (0.6 + 0.4 * depth)
            top = (0.35 + 0.65 * sqrt(depth)) * rng.uniform(0.7, 1.0)
            center = (axes[0][i], axes[1][j], max(top - radius, radius / 2))

            # Only the box around the puff can change, so just work that out
            box = tuple(
                slice(
                    axis.searchsorted(middle - radius),
                    axis.searchsorted(middle + radius, side="right"),
                )
                for axis, middle in zip(axes, center)
            )
            du, dv, dw = (axis[s] - c for axis, s, c in zip(axes, box, center))
            distance = sqrt(
                du[:, None, None] ** 2 + dv[None, :, None] ** 2 + dw[None, None, :] ** 2
            )
            density[box] = maximum(density[box], clip(1 - distance / radius, 0, 1))

        return density

    def surface(
        self, rings: int | None = None, around: int | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        The cloud's smoothed top, as x, y and z on `rings` rings running in from
        its base outline to its centre, with about `around` points on each. The
        outer ring is the outline itself at base height, and the sides round off
        up from it, so top and base meet smoothly all round.

        Both default to as dense as the cloud's own grid, so its `resolution`
        sets how detailed the drawn cloud is.
        """
        width = max(self.density.shape[:2])
        rings = rings if rings is not None else width * 2 // 3
        around = around if around is not None else width * 3

        # Rings shrink the outline towards its centre, bunched up near the edge
        # where the sides curve most
        outline = outline_points(self.base[:, :2], ceil(around / len(self.base)))
        center = self.base[:, :2].mean(axis=0)
        scale = 1 - (1 - linspace(0, 1, rings)) ** 2
        x = center[0] + scale[:, None] * (outline[:, 0] - center[0])
        y = center[1] + scale[:, None] * (outline[:, 1] - center[1])

        # Look each point's height up on the column grid, by its fractional position on it
        grid_x, grid_y = self.x[:, 0, 0], self.y[0, :, 0]
        rows = (x - grid_x[0]) / (grid_x[1] - grid_x[0])
        cols = (y - grid_y[0]) / (grid_y[1] - grid_y[0])
        heights = map_coordinates(self._column_heights(), [rows, cols], order=1)

        # Round the sides off over the outer SIDE_ROUNDING of the way in
        heights *= sqrt(clip((1 - scale[:, None]) / SIDE_ROUNDING, 0, 1))

        return x, y, self.altitude + heights

    def _column_heights(self) -> NDArray[np.float64]:
        """Smoothed height of the cloud's top above its base, over each grid column."""
        cloudy_cells = self.density > CLOUD_EDGE_DENSITY
        cloudy = cloudy_cells.any(axis=2)
        nz = self.density.shape[2]

        # Highest cloudy cell in each column, plus however far up towards the next
        # cell the density takes to drop through the edge value
        highest = nz - 1 - argmax(cloudy_cells[:, :, ::-1], axis=2)
        above = minimum(highest + 1, nz - 1)
        at_highest = take_along_axis(self.density, highest[..., None], axis=2)[..., 0]
        at_above = take_along_axis(self.density, above[..., None], axis=2)[..., 0]
        with errstate(divide="ignore", invalid="ignore"):
            partway = (at_highest - CLOUD_EDGE_DENSITY) / (at_highest - at_above)
        partway = where(above > highest, clip(partway, 0, 1), 0)

        step = self.height / (nz - 1)
        heights = gaussian_filter(
            where(cloudy, (highest + partway) * step, 0),
            sigma=[SMOOTHING * size for size in cloudy.shape],
        )
        # The body stands at least BODY_HEIGHT across the whole base, so the sides
        # are shaped by their rounding, not by where the noise thins out at the edge
        body: NDArray[np.float64] = maximum(heights, BODY_HEIGHT * self.height)
        return body

    def traces(
        self,
        name: str = "Cloud",
        top_color: str = "#ffffff",
        base_color: str = "#e2e6ec",
    ) -> list[Surface]:
        """
        Plotly surfaces drawing the cloud as a solid: its smooth top, shaded from
        `base_color` low down to `top_color` at its peaks, and its flat base. Both
        fade with height, as `fade` sets out.
        """
        # float32 is plenty to draw with (~0.5 m at these coordinates) and keeps the page small
        x, y, tops = (values.astype(float32) for values in self.surface())

        # Opacity follows the same height above the base that the colors do
        rises = linspace(0, 1, 9)
        top = Surface(
            x=x,
            y=y,
            z=tops,
            surfacecolor=tops - float32(self.altitude),
            cmin=0,
            cmax=self.height,
            colorscale=[[0, base_color], [1, top_color]],
            opacityscale=[
                [float(rise), float(opacity)]
                for rise, opacity in zip(rises, self.fade(rises))
            ],
            lighting={
                "ambient": 0.8,
                "diffuse": 0.4,
                "specular": 0.05,
                "roughness": 0.9,
            },
            showscale=False,
            name=name,
            hovertemplate=f"{name}<extra></extra>",
        )
        # Being flat, the base only needs its centre and its outline: the first and last rings
        rim = [0, -1]
        base = solid_surface(
            x[rim],
            y[rim],
            full_like(tops[rim], self.altitude),
            base_color,
            name,
            opacity=float(self.fade(zeros(1))[0]),
        )

        return [top, base]


def bases_overlap(
    first: NDArray[np.float64], second: NDArray[np.float64], gap: float = 0.0
) -> bool:
    """
    Whether two cloud bases (N x 2 or N x 3 polygons; z is ignored) overlap, or
    come within `gap` of each other.
    """
    first, second = first[:, :2], second[:, :2]
    # Points along each outline, close enough together that edges can't cross unseen
    for outline, other in ((first, second), (second, first)):
        points = outline_points(outline, per_edge=4)
        if (signed_distance(other, points[:, 0], points[:, 1]) > -gap).any():
            return True
    return False


def cloud_cover(
    mesh: NDArray[np.floating[Any]],
    amount: CloudAmount,
    altitude: float,
    height: float,
    seed: int | None = None,
    size: tuple[float, float] = (0.06, 0.14),
    resolution: tuple[int, int, int] = CLOUD_RESOLUTION,
    opacity: float = 1.0,
    gap: float = 0.02,
    avoid: Sequence[Cloud] = (),
) -> list[Cloud]:
    """
    Scatter clouds with bases at `altitude` over the terrain in `mesh`, until
    their bases add up to the share of the sky that `amount` reports. Clouds
    never overlap (overlapping clouds would just be one cloud), seen from above
    or below: they keep at least `gap` apart (as a share of the terrain's
    narrower side) from each other, and from the clouds in `avoid`, such as
    other layers'.

    Each cloud is up to `height` tall, with a radius drawn from `size` (also a
    share of the terrain's narrower side), detailed to `resolution`, and as
    opaque as `opacity` at its peaks. Once the sky is too full for clouds that
    size, smaller ones fill the gaps; if even those don't fit, the cover comes
    out short of what `amount` reports.
    """
    x_mesh, y_mesh, _ = mesh
    x_min, x_max = float(x_mesh.min()), float(x_mesh.max())
    y_min, y_max = float(y_mesh.min()), float(y_mesh.max())
    side = min(x_max - x_min, y_max - y_min)
    target = CLOUD_COVERAGE[amount] * (x_max - x_min) * (y_max - y_min)

    rng = default_rng(seed)
    clouds: list[Cloud] = []
    covered, shrink = 0.0, 1.0
    while covered < target and shrink >= MIN_CLOUD_SHRINK:
        base = None
        for _ in range(PLACEMENT_TRIES):
            radius = rng.uniform(*size) * shrink * side
            # Centre it as near the edge as its narrowest wobble allows, then check
            # its actual outline stays over the terrain
            margin = radius * (1 - BASE_WOBBLE)
            center = (
                rng.uniform(x_min + margin, x_max - margin),
                rng.uniform(y_min + margin, y_max - margin),
            )
            candidate = random_base(center, radius, altitude, rng)
            (low_x, low_y), (high_x, high_y) = (
                candidate[:, :2].min(axis=0),
                candidate[:, :2].max(axis=0),
            )
            over_terrain = (
                low_x >= x_min
                and high_x <= x_max
                and low_y >= y_min
                and high_y <= y_max
            )
            if over_terrain and not any(
                bases_overlap(candidate, cloud.base, gap * side)
                for cloud in [*avoid, *clouds]
            ):
                base = candidate
                break

        if base is None:
            shrink *= 0.8  # No room for clouds this size: try smaller ones
            continue

        clouds.append(
            Cloud(
                base,
                height * rng.uniform(0.6, 1.0),
                seed=int(rng.integers(2**32)),
                resolution=resolution,
                opacity=opacity,
            )
        )
        covered += polygon_area(base)

    return clouds
