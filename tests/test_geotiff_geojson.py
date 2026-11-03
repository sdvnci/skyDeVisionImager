from numpy import array, float32, float64, nan
import pytest
from rasterio.warp import transform as warp_transform

from server.geotiff import mesh_to_geojson, xyz_to_geojson


def test_xyz_to_geojson_outputs_multipoint() -> None:
    xyz = array(
        [
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.5],
        ],
        dtype=float32,
    )

    geojson = xyz_to_geojson(xyz)

    assert geojson == {
        "type": "MultiPoint",
        "coordinates": [
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.5],
        ],
    }


def test_mesh_to_geojson_outputs_multipolygon() -> None:
    meshes = array(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=float64,
    )

    geojson = mesh_to_geojson(meshes)

    assert geojson == {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    [0.0, 0.0, 5.0],
                    [1.0, 0.0, 6.0],
                    [1.0, 1.0, 8.0],
                    [0.0, 1.0, 7.0],
                    [0.0, 0.0, 5.0],
                ]
            ]
        ],
    }


def test_mesh_to_geojson_skips_nan_cells() -> None:
    meshes = array(
        [
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[5.0, 6.0, 7.0], [8.0, 9.0, nan]],
        ],
        dtype=float64,
    )

    geojson = mesh_to_geojson(meshes)

    assert geojson == {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    [0.0, 0.0, 5.0],
                    [1.0, 0.0, 6.0],
                    [1.0, 1.0, 9.0],
                    [0.0, 1.0, 8.0],
                    [0.0, 0.0, 5.0],
                ]
            ]
        ],
    }


def test_xyz_to_geojson_reprojects_projected_coordinates() -> None:
    xyz = array(
        [
            [315000.0, 6400000.0, 30.0],
            [317500.0, 6397500.0, 31.5],
        ],
        dtype=float32,
    )

    expected_xs, expected_ys = warp_transform(
        "EPSG:3006",
        "EPSG:4326",
        [315000.0, 317500.0],
        [6400000.0, 6397500.0],
    )

    geojson = xyz_to_geojson(xyz, source_crs="EPSG:3006")

    assert geojson["type"] == "MultiPoint"
    coordinates = geojson["coordinates"]
    assert coordinates[0][0] == pytest.approx(expected_xs[0])
    assert coordinates[0][1] == pytest.approx(expected_ys[0])
    assert coordinates[0][2] == pytest.approx(30.0)
    assert coordinates[1][0] == pytest.approx(expected_xs[1])
    assert coordinates[1][1] == pytest.approx(expected_ys[1])
    assert coordinates[1][2] == pytest.approx(31.5)


def test_mesh_to_geojson_requires_wgs84_or_source_crs() -> None:
    meshes = array(
        [
            [[315000.0, 317500.0], [315000.0, 317500.0]],
            [[6400000.0, 6400000.0], [6397500.0, 6397500.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=float64,
    )

    with pytest.raises(ValueError, match="GeoJSON requires WGS84"):
        mesh_to_geojson(meshes)
