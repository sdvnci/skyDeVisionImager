"""
Example scene: the terrain on a solid block of ground, with layers of fluffy,
solid clouds scattered above it. Writes the interactive plot to topo.html.
"""

import time
from json import dumps
from pathlib import Path
from typing import NamedTuple

from numpy import float32, nanmax, uint16
from plotly.graph_objects import Figure  # type: ignore

from server.geotiff import (
    Cloud,
    CloudAmount,
    cloud_cover,
    fit_scene_height,
    graph_plotly,
    ground_block,
    mesh_to_geojson,
    tif_to_mesh,
)

BASEPATH = Path(__file__).parent / "hojddata" / "63_3"
TFWFILE = BASEPATH / "639_31_7550_2019.tfw"
TIFFFILE = BASEPATH / "639_31_7550_2019.tif"

GROUND_COLOR = "#7a5230"
GROUND_DEPTH = 0.3  # how far the block's base sits below the terrain's lowest point


class CloudLayer(NamedTuple):
    amount: CloudAmount  # METAR cloud amount: the share of the sky it covers
    altitude: float  # base height above the highest peak (terrain z runs 0..1)
    height: float  # tallest cloud
    opacity: float  # at the clouds' peaks; they fade towards their bases from there
    size: tuple[float, float]  # range of cloud radii, as a share of the terrain's width


# Bottom to top. No cloud overlaps another, even in other layers, as seen from
# below, so higher layers fill in the gaps left by lower ones.
CLOUD_LAYERS = (
    CloudLayer("SCT", altitude=0.6, height=0.5, opacity=1.0, size=(0.14, 0.22)),
    CloudLayer("FEW", altitude=1.3, height=0.35, opacity=0.7, size=(0.08, 0.12)),
)
CLOUD_RESOLUTION = (96, 96, 48)  # cloud detail: noise grid points along x, y and z
CLOUD_SEED = 7  # change for a different sky

Z_SCALE = 0.35  # scene height per unit of z
WRITE_GEOJSON = False  # also export the terrain mesh as GeoJSON

start_time = time.time()

mesh = tif_to_mesh(tfwfile=TFWFILE, tifffile=TIFFFILE, points=uint16(100))

figure = Figure()

# Ground first, so the semitransparent terrain is drawn over it
figure.add_traces(ground_block(mesh, depth=GROUND_DEPTH, color=GROUND_COLOR))
graph_plotly(meshes=mesh, cmapstr="solar", opacity=float32(0.75), figure=figure)

terrain_top = float(nanmax(mesh[2]))
sky: list[Cloud] = []  # every layer's clouds so far, so later layers keep clear of them
for index, layer in enumerate(CLOUD_LAYERS):
    clouds = cloud_cover(
        mesh,
        amount=layer.amount,
        altitude=terrain_top + layer.altitude,
        height=layer.height,
        seed=CLOUD_SEED + index,
        size=layer.size,
        resolution=CLOUD_RESOLUTION,
        opacity=layer.opacity,
        avoid=sky,
    )
    sky.extend(clouds)
    for cloud in clouds:
        figure.add_traces(cloud.traces(name=f"{layer.amount} cloud"))

fit_scene_height(figure, Z_SCALE)

processing_time = time.time() - start_time
print(f"Processing completed in {processing_time:.2f} seconds")

figure.write_html("topo.html", full_html=True, include_plotlyjs=True)
print("HTML file generated successfully: topo.html")

if WRITE_GEOJSON:
    geojson = mesh_to_geojson(mesh, source_crs="EPSG:3006")
    with (BASEPATH / "geojson_example.json").open("w") as f:
        f.write(dumps(geojson, indent=2))
