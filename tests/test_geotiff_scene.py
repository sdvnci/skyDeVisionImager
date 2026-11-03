import os
from contextlib import nullcontext
from pathlib import Path

from numpy import array, float32, float64, isfinite, meshgrid, linspace, uint16, zeros
from numpy.random import default_rng
from numpy.typing import NDArray
import pytest

from server.geotiff import Cloud, cloud_cover, ground_block, random_base, tif_to_mesh
from server.geotiff import topography
from server.geotiff.clouds import (
    bases_overlap,
    CLOUD_COVERAGE,
    CLOUD_EDGE_DENSITY,
    CLOUD_FADE,
    polygon_area,
    signed_distance,
)

SQUARE = array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float64)


def flat_mesh(size: float = 1000.0, points: int = 20) -> NDArray[float64]:
    x_mesh, y_mesh = meshgrid(linspace(0, size, points), linspace(0, size, points))
    z_mesh = (x_mesh + y_mesh) / (2 * size)
    return array([x_mesh, y_mesh, z_mesh], dtype=float64)


def square_base(altitude: float = 2.0) -> NDArray[float64]:
    return array([[x * 100, y * 100, altitude] for x, y in SQUARE], dtype=float64)


def test_signed_distance_is_positive_inside_and_negative_outside() -> None:
    distance = signed_distance(SQUARE, array([0.5, 0.25, 1.5]), array([0.5, 0.5, 0.5]))

    assert distance == pytest.approx([0.5, 0.25, -0.5])


def test_polygon_area_of_unit_square() -> None:
    assert polygon_area(SQUARE) == pytest.approx(1.0)


def test_random_base_is_flat_and_wobbles_around_its_radius() -> None:
    base = random_base((10.0, 20.0), radius=5.0, altitude=3.0, rng=default_rng(1))

    distances = ((base[:, 0] - 10.0) ** 2 + (base[:, 1] - 20.0) ** 2) ** 0.5
    assert (base[:, 2] == 3.0).all()
    assert distances.min() >= 5.0 * 0.7 - 1e-9
    assert distances.max() <= 5.0 * 1.3 + 1e-9


@pytest.mark.parametrize(
    ("base", "height", "message"),
    [
        (square_base()[:2], 1.0, "at least 3"),
        (array([[0, 0, 0], [1, 0, 0], [1, 1, 0.5]], dtype=float64), 1.0, "flat"),
        # A bow-tie: its outline crosses back over itself around the centre
        (
            array([[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=float64),
            1.0,
            "star-shaped",
        ),
        (square_base(), 0.0, "positive"),
    ],
)
def test_cloud_rejects_invalid_shapes(
    base: NDArray[float64], height: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Cloud(base, height)


@pytest.mark.parametrize("opacity", [0.0, -0.5, 1.5])
def test_cloud_rejects_opacity_outside_0_to_1(opacity: float) -> None:
    with pytest.raises(ValueError, match="opacity"):
        Cloud(square_base(), height=0.5, opacity=opacity)


def test_cloud_has_a_flat_base_on_its_outline() -> None:
    cloud = Cloud(square_base(altitude=2.0), height=0.5, seed=3)

    assert cloud.altitude == 2.0
    assert cloud.z.min() == pytest.approx(2.0)
    assert cloud.z.max() == pytest.approx(2.5)

    # At the base, there's cloud in the middle of the outline and clear air outside it
    base_slice = cloud.density[:, :, 0]
    outside = signed_distance(SQUARE * 100, cloud.x[:, :, 0], cloud.y[:, :, 0]) < 0
    middle = base_slice.shape[0] // 2, base_slice.shape[1] // 2
    assert base_slice[middle] > CLOUD_EDGE_DENSITY
    assert (base_slice[outside] == 0).all()

    # And nothing reaches the top of its grid, so the top is never cut flat
    assert (cloud.density[:, :, -1] < CLOUD_EDGE_DENSITY).all()


def test_cloud_is_repeatable_with_a_seed() -> None:
    first = Cloud(square_base(), height=0.5, seed=11)
    again = Cloud(square_base(), height=0.5, seed=11)
    other = Cloud(square_base(), height=0.5, seed=12)

    assert (first.density == again.density).all()
    assert not (first.density == other.density).all()


def test_cloud_top_comes_down_to_meet_its_base_on_the_outline() -> None:
    cloud = Cloud(square_base(altitude=2.0), height=0.5, seed=1)

    x, y, z = cloud.surface()

    assert isfinite(z).all()
    assert z.min() == pytest.approx(2.0)
    assert z.max() <= 2.5
    assert z.max() > 2.0 + 0.25  # it does rise, somewhere

    # The outer ring sits at base height, exactly on the base outline, closing the solid
    assert z[-1] == pytest.approx(2.0)
    assert signed_distance(SQUARE * 100, x[-1], y[-1]) == pytest.approx(0, abs=1e-9)
    # ...and every other ring is inside it
    assert (signed_distance(SQUARE * 100, x[:-1], y[:-1]) > 0).all()


def test_cloud_traces_are_a_smooth_top_and_a_flat_base() -> None:
    cloud = Cloud(square_base(altitude=2.0), height=0.5, seed=1)

    top, base = cloud.traces(name="SCT cloud")

    assert top.type == base.type == "surface"
    assert top.name == base.name == "SCT cloud"
    assert isfinite(top.z).all()
    # The base is flat, and runs out to exactly where the top comes down to it
    assert base.z == pytest.approx(2.0)
    assert (base.x[-1] == top.x[-1]).all()
    assert (base.y[-1] == top.y[-1]).all()


@pytest.mark.parametrize("opacity", [1.0, 0.6])
def test_cloud_fades_from_its_base_up_to_its_opacity(opacity: float) -> None:
    cloud = Cloud(square_base(), height=0.5, seed=1, opacity=opacity)

    top, base = cloud.traces()

    rises, opacities = zip(*top.opacityscale)
    assert rises[0] == 0 and rises[-1] == 1
    assert opacities[0] == pytest.approx(CLOUD_FADE * opacity)
    assert opacities[-1] == pytest.approx(opacity)
    assert list(opacities) == sorted(opacities)  # only ever more solid going up
    # The flat base is as faded as the bottom of the top surface
    assert base.opacity == pytest.approx(CLOUD_FADE * opacity)


def test_cloud_cover_passes_opacity_to_its_clouds() -> None:
    clouds = cloud_cover(
        flat_mesh(), "FEW", 2.0, 0.4, seed=5, resolution=(16, 16, 8), opacity=0.7
    )

    assert {cloud.opacity for cloud in clouds} == {0.7}


def test_cloud_surface_detail_follows_its_resolution() -> None:
    coarse = Cloud(square_base(), height=0.5, seed=1, resolution=(24, 24, 12))
    fine = Cloud(square_base(), height=0.5, seed=1, resolution=(72, 72, 36))

    coarse_x, *_ = coarse.surface()
    fine_x, *_ = fine.surface()

    assert fine.density.shape == (72, 72, 36)
    assert fine_x.size > 4 * coarse_x.size
    # Asked-for density still wins over the default (4 sides of 10, closed back to the start)
    custom_x, *_ = fine.surface(rings=10, around=40)
    assert custom_x.shape == (10, 40 + 1)


def test_bases_overlap_when_they_cross_or_come_within_the_gap() -> None:
    base = square_base()  # 0..100 square
    shifted = base + array([150.0, 0.0, 0.0])  # 50 clear of it
    inside = base * array([0.5, 0.5, 1.0]) + array([25.0, 25.0, 0.0])

    assert bases_overlap(base, base + array([50.0, 50.0, 0.0]))
    assert bases_overlap(base, inside)  # wholly inside, so no edges cross
    assert not bases_overlap(base, shifted)
    assert not bases_overlap(base, shifted, gap=40.0)
    assert bases_overlap(base, shifted, gap=60.0)


def test_cloud_cover_reaches_the_reported_share_of_sky_without_overlap() -> None:
    mesh = flat_mesh()

    clouds = cloud_cover(
        mesh, "FEW", altitude=2.0, height=0.4, seed=5, resolution=(16, 16, 8)
    )

    covered = sum(polygon_area(cloud.base) for cloud in clouds)
    assert covered >= CLOUD_COVERAGE["FEW"] * 1000.0**2
    for index, cloud in enumerate(clouds):
        assert cloud.density.shape == (16, 16, 8)
        assert cloud.altitude == 2.0
        assert cloud.height <= 0.4
        assert cloud.base[:, :2].min() >= 0.0
        assert cloud.base[:, :2].max() <= 1000.0
        assert not any(
            bases_overlap(cloud.base, other.base) for other in clouds[:index]
        )


def test_cloud_cover_keeps_clear_of_clouds_it_is_told_to_avoid() -> None:
    lower = cloud_cover(flat_mesh(), "SCT", 2.0, 0.4, seed=5, resolution=(8, 8, 4))

    upper = cloud_cover(
        flat_mesh(), "FEW", 3.0, 0.4, seed=6, resolution=(8, 8, 4), avoid=lower
    )

    assert upper
    for cloud in upper:
        assert not any(bases_overlap(cloud.base, other.base) for other in lower)


def test_cloud_cover_stops_short_rather_than_overlap() -> None:
    # Overcast can't be tiled by separate round clouds, so it fills what it can
    clouds = cloud_cover(
        flat_mesh(), "OVC", altitude=2.0, height=0.4, seed=5, resolution=(8, 8, 4)
    )

    covered = sum(polygon_area(cloud.base) for cloud in clouds)
    assert 0.3 * 1000.0**2 < covered < 1000.0**2
    for index, cloud in enumerate(clouds):
        assert not any(
            bases_overlap(cloud.base, other.base) for other in clouds[:index]
        )


def test_ground_block_walls_run_from_terrain_edge_to_flat_base() -> None:
    mesh = flat_mesh(points=5)

    walls, floor = ground_block(mesh, depth=0.3, color="#7a5230")

    top, bottom = walls.z
    assert top.min() == pytest.approx(0.0)
    assert top.max() == pytest.approx(1.0)
    assert bottom == pytest.approx(-0.3)
    assert floor.z == pytest.approx(-0.3)
    assert len(top) == 4 * (5 - 1) + 1  # once around the edge, back to the first corner


def test_tif_to_mesh_reuses_its_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tfwfile, tifffile = tmp_path / "tile.tfw", tmp_path / "tile.tif"
    tfwfile.write_text("1\n0\n0\n-1\n0\n0\n")
    tifffile.write_bytes(b"")

    builds: list[int] = []

    def fake_meshify(xyz: object, **_: object) -> NDArray[float64]:
        builds.append(1)
        return flat_mesh(points=3)

    monkeypatch.setattr(topography, "rio_open", lambda _: nullcontext())
    monkeypatch.setattr(topography, "tfw_to_xyz", lambda *_: zeros((4, 3), float32))
    monkeypatch.setattr(topography, "meshify", fake_meshify)

    first = tif_to_mesh(tfwfile, tifffile, cachedir=tmp_path, points=uint16(3))
    second = tif_to_mesh(tfwfile, tifffile, cachedir=tmp_path, points=uint16(3))

    assert len(builds) == 1
    assert (first == second).all()

    # Changing the GeoTIFF invalidates the cache
    modified = tifffile.stat().st_mtime_ns + 1_000_000_000
    os.utime(tifffile, ns=(modified, modified))
    tif_to_mesh(tfwfile, tifffile, cachedir=tmp_path, points=uint16(3))
    assert len(builds) == 2
