"""Virtual-species demo data.

No real species data is needed to run the pipeline end to end. This module
builds a plausible, *synthetic* Mediterranean-style environment and a "virtual
species" whose true niche is known (the approach popularised by the R package
``virtualspecies``). Because the truth is known, the demo can check that the
pipeline recovers it. The numbers it produces say nothing about the real
distribution of *Mentha pulegium*.

Source rasters are written at deliberately different resolutions and with a
buffer beyond the study area, so the clip + resample logic in stage 2 is
genuinely exercised.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom

from .grid import Grid
from .io import write_raster

BUFFER_DEG = 1.0
# (file, layer) at their native source resolution factors relative to the base grid
DEM_UPSAMPLE = 2  # DEM at half the base cell size (finer)
CLIMATE_COARSEN = 2  # climate at twice the base cell size
SOIL_COARSEN = 4  # soil at four times the base cell size


def _smooth_noise(rng, shape, sigma):
    n = gaussian_filter(rng.normal(size=shape), sigma, mode="reflect")
    return (n - n.mean()) / n.std()


def _gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _block_mean(a: np.ndarray, k: int) -> np.ndarray:
    h, w = a.shape
    return a[: h - h % k, : w - w % k].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def make_environment(grid: Grid, seed: int) -> dict[str, np.ndarray]:
    """Build the (buffered) base fields on ``grid``."""
    rng = np.random.default_rng(seed)
    h, w = grid.shape
    lon, lat = grid.cell_centres()
    lon_n = (lon - grid.west) / (grid.east - grid.west)
    lat0 = grid.south

    ridge = _smooth_noise(rng, (h, w), sigma=h / 10)
    detail = _smooth_noise(rng, (h, w), sigma=h / 35)
    bump = np.exp(-(((lon_n - 0.5) / 0.28) ** 2 + ((lat - (lat0 + 0.45 * (grid.north - lat0))) / 2.0) ** 2))
    elevation = np.clip(450 + 380 * ridge + 130 * detail + 650 * bump, 0, 3200)

    t_noise = _smooth_noise(rng, (h, w), sigma=h / 15)
    bio1 = 20.5 - 0.55 * (lat - lat0 - 1.0) - 0.0065 * elevation + 0.7 * t_noise  # mean annual temp, degC

    p_noise = _smooth_noise(rng, (h, w), sigma=h / 12)
    bio12 = np.clip(620 + 330 * p_noise + 0.22 * elevation + 160 * (1 - lon_n), 150, 2300)  # annual precip, mm

    s_noise = _smooth_noise(rng, (h, w), sigma=h / 14)
    bio15 = np.clip(52 + 11 * s_noise - 0.012 * (bio12 - 620), 15, 95)  # precip seasonality, CV %

    ndvi_true = np.clip(
        0.18 + 0.50 * np.tanh(bio12 / 900.0) - 0.012 * np.maximum(bio1 - 16, 0) ** 1.4 + 0.04 * _smooth_noise(rng, (h, w), 3),
        0.05,
        0.9,
    )

    ph = np.clip(6.9 + 0.7 * _smooth_noise(rng, (h, w), h / 11) - 0.0004 * (bio12 - 620), 4.5, 8.5)
    clay = np.clip(26 + 10 * _smooth_noise(rng, (h, w), h / 13), 5, 60)

    return {
        "elevation": elevation,
        "bio1": bio1,
        "bio12": bio12,
        "bio15": bio15,
        "ndvi_true": ndvi_true,
        "soil_ph": ph,
        "clay": clay,
    }


def _raw_suitability(env: dict[str, np.ndarray], slope: np.ndarray) -> np.ndarray:
    """The hidden 'true' niche: moist, temperate, low-to-mid elevation, near-neutral soils."""
    return (
        _gauss(env["bio1"], 13.5, 3.0)
        * _gauss(env["bio12"], 900, 330)
        * _gauss(env["elevation"], 500, 600)
        * _gauss(env["soil_ph"], 6.8, 0.9)
        * _gauss(env["bio15"], 55, 22)
        * _gauss(env["ndvi_true"], 0.55, 0.25)
        * _gauss(slope, 2.0, 5.0)
        * (0.5 + 0.5 * _sigmoid((env["clay"] - 20) / 8))
    )


def virtual_species_suitability(env: dict[str, np.ndarray], slope: np.ndarray, norm: float | None = None) -> np.ndarray:
    """True suitability in [0, 1]. ``norm`` fixes the scaling so other regions share the same scale."""
    s = _raw_suitability(env, slope)
    if norm is None:
        norm = float(np.nanpercentile(s, 99.5))
    return np.clip(s / norm, 0, 1).astype("float32")


def _write_scene(path, ndvi, rng, grid, cloud_frac):
    """Fake Sentinel-2-like scene: red/NIR reflectance x 10000 plus an SCL band with clouds."""
    nir = np.clip(rng.normal(0.38, 0.03, ndvi.shape) + 0.25 * ndvi, 0.05, 0.9)
    nd = np.clip(ndvi + rng.normal(0, 0.03, ndvi.shape), -0.2, 0.95)
    red = np.clip(nir * (1 - nd) / (1 + nd), 0.005, 0.9)
    scl = np.full(ndvi.shape, 4, dtype="uint16")  # 4 = vegetation
    cloud_noise = gaussian_filter(rng.normal(size=ndvi.shape), 5)
    clouds = cloud_noise > np.quantile(cloud_noise, 1 - cloud_frac)
    scl[clouds] = 9
    shadow = np.roll(clouds, (4, 6), axis=(0, 1)) & ~clouds
    scl[shadow] = 3
    # cloud pixels are bright and flat in both bands -> NDVI ~ 0 if left unmasked
    red = np.where(clouds, 0.55, red)
    nir = np.where(clouds, 0.58, nir)
    data = np.stack([(red * 10000).astype("uint16"), (nir * 10000).astype("uint16"), scl])
    write_raster(path, data, grid, nodata=None, dtype="uint16", descriptions=["red", "nir", "scl"])


def _write_source_layers(env: dict[str, np.ndarray], big: Grid, layers_dir: Path) -> None:
    """Write source rasters at deliberately different resolutions (exercises clip + resample)."""
    dem_grid = big.with_resolution(big.resolution / DEM_UPSAMPLE)
    dem = zoom(env["elevation"], DEM_UPSAMPLE, order=1).astype("float32")
    write_raster(layers_dir / "elevation_dem.tif", dem, dem_grid)

    clim_grid = big.with_resolution(big.resolution * CLIMATE_COARSEN)
    for name in ("bio1", "bio12", "bio15"):
        write_raster(layers_dir / f"{name}.tif", _block_mean(env[name], CLIMATE_COARSEN), clim_grid)

    soil_grid = big.with_resolution(big.resolution * SOIL_COARSEN)
    for name in ("soil_ph", "clay"):
        write_raster(layers_dir / f"{name}.tif", _block_mean(env[name], SOIL_COARSEN), soil_grid)


def generate_demo_data(study: Grid, out_dir, seed: int = 42, n_presences: int = 350, n_scenes: int = 5):
    """Write source rasters + dirty occurrence table + hidden truth. Returns the truth surface (study grid)."""
    out_dir = Path(out_dir)
    layers_dir = out_dir / "source_layers"
    scenes_dir = out_dir / "s2_scenes"
    rng = np.random.default_rng(seed)

    big = study.buffered(BUFFER_DEG)
    env = make_environment(big, seed)

    # slope needs the DEM, computed here on the buffered base grid for the truth only
    from .processing import derive_slope

    slope_big = derive_slope(env["elevation"], big)
    norm = float(np.nanpercentile(_raw_suitability(env, slope_big), 99.5))
    truth_big = virtual_species_suitability(env, slope_big, norm=norm)

    # --- crop truth to the study area
    r0 = int(round((big.north - study.north) / big.resolution))
    c0 = int(round((study.west - big.west) / big.resolution))
    truth = truth_big[r0 : r0 + study.height, c0 : c0 + study.width]

    _write_source_layers(env, big, layers_dir)

    # --- cloudy "Sentinel-2" scenes
    for i in range(n_scenes):
        _write_scene(scenes_dir / f"scene_{i+1:02d}.tif", env["ndvi_true"], rng, big, cloud_frac=rng.uniform(0.12, 0.35))

    # --- presences sampled in proportion to true suitability
    h, w = study.shape
    weights = (truth.astype("float64") ** 2).ravel()
    weights /= weights.sum()
    cells = rng.choice(h * w, size=n_presences, replace=False, p=weights)
    rows, cols = np.divmod(cells, w)
    lon = study.west + (cols + rng.random(n_presences)) * study.resolution
    lat = study.north - (rows + rng.random(n_presences)) * study.resolution
    occ = pd.DataFrame(
        {
            "gbifID": np.arange(1_000_000, 1_000_000 + n_presences),
            "species": "Mentha pulegium",
            "lon": lon,
            "lat": lat,
            "coordinateUncertaintyInMeters": rng.choice([10, 30, 100, 500, 1000, np.nan], n_presences, p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.2]),
            "year": rng.integers(1985, 2025, n_presences),
            "basisOfRecord": rng.choice(["HUMAN_OBSERVATION", "PRESERVED_SPECIMEN"], n_presences, p=[0.8, 0.2]),
            "datasetKey": "demo-virtual-species",
            "countryCode": "XX",
        }
    )
    occ = _add_dirty_records(occ, study, rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    occ.to_csv(out_dir / "occurrences_raw.csv", index=False)
    write_raster(out_dir / "virtual_species_truth.tif", truth, study)
    (out_dir / "truth_norm.json").write_text(json.dumps({"norm": norm}))
    return truth


def _add_dirty_records(occ: pd.DataFrame, study: Grid, rng) -> pd.DataFrame:
    """Append realistic data-quality problems so the cleaning step has real work to do."""
    n = len(occ)
    pieces = [occ]
    base = occ.iloc[:1].copy()

    def dup(k, **over):
        d = occ.sample(k, random_state=int(rng.integers(0, 10_000))).copy()
        for key, val in over.items():
            d[key] = val
        return d

    missing = dup(14)
    missing["lon"], missing["lat"] = np.nan, np.nan
    zero = dup(6)
    zero["lon"], zero["lat"] = 0.0, 0.0
    outside = dup(10)
    outside["lon"] = study.east + rng.uniform(1, 8, len(outside))
    coarse = dup(12, coordinateUncertaintyInMeters=25000.0)
    old = dup(8, year=1962)
    exact_dups = occ.sample(18, random_state=7).copy()
    near_dups = occ.sample(25, random_state=11).copy()
    near_dups["lon"] += rng.normal(0, study.resolution * 0.15, len(near_dups))
    near_dups["lat"] += rng.normal(0, study.resolution * 0.15, len(near_dups))
    pieces += [missing, zero, outside, coarse, old, exact_dups, near_dups]
    out = pd.concat(pieces, ignore_index=True)
    out["gbifID"] = np.arange(1_000_000, 1_000_000 + len(out))
    return out.sample(frac=1, random_state=int(rng.integers(0, 10_000))).reset_index(drop=True)


def generate_target_region(
    target: Grid, out_dir, seed: int = 42, warm: float = 5.0, wet: float = 350.0
):
    """Synthetic *target* region for the transfer demo, with a shifted climate.

    Same virtual species, same niche, same truth scale as the training region, but the
    region is warmer (+``warm`` degC) and wetter (+``wet`` mm) so that much of it lies
    outside the training climate. Returns the true suitability on the target grid.
    """
    out_dir = Path(out_dir)
    norm = json.loads((out_dir / "truth_norm.json").read_text())["norm"]
    big = target.buffered(BUFFER_DEG)
    env = make_environment(big, seed + 1000)
    env["bio1"] = env["bio1"] + warm
    env["bio12"] = np.clip(env["bio12"] + wet, 150, 3200)

    from .processing import derive_slope

    slope_big = derive_slope(env["elevation"], big)
    truth_big = virtual_species_suitability(env, slope_big, norm=norm)
    r0 = int(round((big.north - target.north) / big.resolution))
    c0 = int(round((target.west - big.west) / big.resolution))
    truth = truth_big[r0 : r0 + target.height, c0 : c0 + target.width]
    _write_source_layers(env, big, out_dir / "target_layers")
    write_raster(out_dir / "target_truth.tif", truth, target)
    return truth
