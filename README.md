# SkyDeVision Imaging: Advanced Environmental Monitoring & Computer Vision Platform

![Unit Tests](https://github.com/sudoDeVinci/skyDeVisionImager/actions/workflows/testing.yml/badge.svg)
![Linting](https://github.com/sudoDeVinci/skyDeVisionImager/actions/workflows/linting.yml/badge.svg)
![Type Check](https://github.com/sudoDeVinci/skyDeVisionImager/actions/workflows/typecheck.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
<a href="https://lnu.diva-portal.org/smash/record.jsf?pid=diva2%3A1874520&dswid=4867"><img src="https://img.shields.io/badge/Paper-DiVA.org.lnu:130812-Green"></a>

![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Validation: Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)

Environmental monitoring and computer vision platform that combines terrestrial imaging analysis with geospatial data processing. Built around a Flask REST API, it provides tools for cloud detection, terrain visualization, camera calibration, and environmental data management from ESP32-based weather stations.

![Image recognition thumbnail](thumbnail.png)

## Overview

### Server
Flask-based REST API handling device registration, environmental readings, and sensor status updates from ESP32 weather stations. Includes automatic METAR data caching for QNH retrieval.

### Image Analysis
Multi-color-space cloud detection (RGB/HSV/YCrCb) with ROC curve analysis and Jaccard similarity metrics. Supports camera calibration and mask generation.

### Geospatial Processing
Numba-accelerated GeoTIFF processing for 3D terrain visualization with Plotly.

## Quick Start

### Installation
```bash
git clone https://github.com/sudoDeVinci/skyDeVisionImager.git
cd skyDeVisionImager
pip install -r requirements.txt
```

### Configuration
Settings are read from the environment, or a `.env` file:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DBSECRET` | Secret the server signs sessions and peppers passwords with. **Required.** | - |
| `DBFILE` | Path to the SQLite database file | `database.db` |
| `DB_POOL_SIZE` | Most database connections open at once | `5` |
| `DB_POOL_TIMEOUT` | Seconds to wait for a free connection, or a locked database | `10` |
| `METAR-TAF` | metar-taf.com API key. Without it the QNH cache is never refreshed. | - |

### Development
```bash
black --check .   # formatting
mypy .            # strict type checking
pytest -q         # tests
```

## Server Components

### API Endpoints (`server/api.py`)

**Device Management**
- `POST /api/register` - Register new weather station (requires MAC address, device/camera model, firmware version, location)
- `POST /api/status` - Update sensor status (SHT, BMP, CAM, WIFI)

**Data Collection**
- `POST /api/reading` - Submit environmental reading (temperature, humidity, pressure, dewpoint). The reading is stamped with `X-Timestamp`; resending one replaces it.
- `GET /api/qnh` - Retrieve current QNH from cached METAR data. `?airport=` picks one of the cached airports.

**System**
- `GET /api/check` - API health check

Device endpoints (`register`, `status`, `reading`) expect headers: `X-MAC-Address`, `X-Timestamp`, `X-Firmware-Version`

Errors are returned as `{"errors": [{"title", "details", "code", "source"}], "timestamp"}`.

### Database (`server/db/`)

**Schema**
- `Stations` - Device registry (MAC, name, device/camera model, firmware, coordinates, altitude)
- `Readings` - Environmental data time series (MAC, timestamp, temp, humidity, pressure, dewpoint, filepath)
- `Status` - Sensor health tracking (MAC, timestamp, SHT, BMP, CAM, WIFI flags)
- `Users` - Authentication (ID, name, email, bcrypt-hashed passwords, role)
- `Locations` - Geographic reference data

**Services**
- `StationService` - CRUD operations for weather stations
- `ReadingService` - Environmental data management
- `StatusService` - Sensor status tracking
- `UserService` - User management with password hashing

Services are never instantiated; every method is a classmethod. Each call is its own
transaction unless it is handed a `cursor`, which is how operations across tables stay atomic:

```python
from server.db import Manager, ReadingService, StationService

with Manager.cursor() as cursor:  # commits on success, rolls back on any error
    StationService.insert(station, cursor=cursor)
    ReadingService.insert(reading, cursor=cursor)
```

### METAR Integration (`server/metar/`)

**MetarCacheLayer**
- Background thread updates QNH values every 10 minutes via metar-taf.com API, once `start()` is called
- Thread-safe cache, saved to disk so a restart serves the last known values
- Supports multiple airport codes
- Graceful fallback on API failures

### Image Analysis (`server/imageanalysis/`)

**Extraction** (`extraction.py`)
- Non-black pixel extraction across RGB/HSV/YCrCb color spaces
- Frequency distribution analysis
- Sparse and full image stacking utilities

**ROC Analysis** (`roccurve.py`)
- Bootstrap sampling for statistical validation
- Jaccard similarity computation for cloud/sky classification
- Channel-wise ROC curve generation

### Geospatial Processing (`server/geotiff/`)

**Topography** (`topography.py`)
- Numba JIT-compiled coordinate extraction from GeoTIFF/TFW files
- Median and Gaussian filtering for denoising
- Plotly 3D surface plots
- Optional downsampling for large datasets

### Running the Server
```python
from server import Manager, create_app

# Loads the configuration and starts the QNH cache
app = create_app()

# Initialize database
Manager.connect("weather_data.db")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### GeoTIFF Visualization
```python
from server.geotiff import graph_plotly
from pathlib import Path

figure = graph_plotly(
    tfwfile=Path("data/terrain.tfw"),
    tifffile=Path("data/terrain.tif"),
    points=200,
    use_cache=True,
)

figure.write_html("terrain.html")
```

### GeoJSON Export
```python
from server.geotiff import meshify, mesh_to_geojson, tfw_to_xyz, xyz_to_geojson
from rasterio import open as rio_open

with open("data/terrain.tfw", "r") as tfw:
    with rio_open("data/terrain.tif") as tif:
        xyz = tfw_to_xyz(tfw, tif)

point_cloud = xyz_to_geojson(xyz, source_crs=tif.crs)          # GeoJSON MultiPoint
surface = mesh_to_geojson(meshify(xyz), source_crs=tif.crs)    # GeoJSON MultiPolygon
```

### ROC Analysis for Cloud Detection
```python
from server.imageanalysis import Camera, ColourTag, ROCAnalyzer, AnalysisConfiguration
from server.db import CameraModel
from numpy import uint16, uint8, float32

camera = Camera(CameraModel.DSLR)
config = AnalysisConfiguration(
    strata_count=uint16(10),
    strata_size=uint16(10),
    boundary_width=uint8(5),
    jaccard_threshold=float32(0.15),
    max_workers=uint8(2),
    caching=True,
    gpu_caching=False
)

analyzer = ROCAnalyzer(camera=camera, config=config)
results = analyzer.analyze_roc(
    camera=camera,
    colortags=[ColourTag.HSV, ColourTag.RGB, ColourTag.YBR]
)
```

### Registering a Weather Station
```bash
curl -X POST http://localhost:5000/api/register \
  -H "X-MAC-Address: AA:BB:CC:DD:EE:FF" \
  -H "X-Timestamp: 2024-01-01T12:00:00Z" \
  -H "X-Firmware-Version: 1.0.0" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Station Alpha",
    "device_model": "ESP32S3",
    "camera_model": "OV5640",
    "altitude": 150.5,
    "latitude": 56.8796,
    "longitude": 14.8094
  }'
```

### Submitting Environmental Readings
```bash
curl -X POST http://localhost:5000/api/reading \
  -H "X-MAC-Address: AA:BB:CC:DD:EE:FF" \
  -H "X-Timestamp: 2024-01-01T12:00:00Z" \
  -H "X-Firmware-Version: 1.0.0" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 22.5,
    "humidity": 65.0,
    "pressure": 1013.25,
    "dewpoint": 15.3
  }'
```

## License

MIT

## Citation

If you use this project in research, please cite:
```
@misc{skydevisionimager,
  title={SkyDeVision Imaging: Advanced Environmental Monitoring & Computer Vision Platform},
  url={https://lnu.diva-portal.org/smash/record.jsf?pid=diva2%3A1874520},
  year={2024}
}
```
