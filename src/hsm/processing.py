"""Stage 2 - data processing.

* occurrence cleaning and spatial thinning
* Sentinel-2 cloud masking and NDVI compositing
* clipping / resampling every layer onto the common grid
* terrain derivatives (slope)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

from .grid import Grid

log = logging.getLogger(__name__)

# Sentinel-2 L2A Scene Classification Layer classes treated as unusable:
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8/9 cloud (medium/high prob.),
# 10 thin cirrus, 11 snow/ice.
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)


# --------------------------------------------------------------------------- #
# Occurrence cleaning
# --------------------------------------------------------------------------- #
def clean_occurrences(
    df: pd.DataFrame,
    grid: Grid,
    max_uncertainty_m: float | None = 5000,
    year_min: int | None = None,
    thin_to_grid: bool = True,
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Apply standard presence-record hygiene.

    Returns the cleaned frame and a list of ``(step, records_remaining)`` so the
    attrition can be reported and plotted.
    """
    steps: list[tuple[str, int]] = [("raw records", len(df))]
    out = df.copy()

    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out = out.dropna(subset=["lon", "lat"])
    steps.append(("has coordinates", len(out)))

    valid = out["lon"].between(-180, 180) & out["lat"].between(-90, 90)
    zero = (out["lon"].abs() < 1e-6) & (out["lat"].abs() < 1e-6)  # the classic (0, 0) error
    out = out[valid & ~zero]
    steps.append(("valid, non-(0,0) coordinates", len(out)))

    out = out[grid.contains(out["lon"].to_numpy(), out["lat"].to_numpy())]
    steps.append(("inside study area", len(out)))

    if max_uncertainty_m is not None and "coordinateUncertaintyInMeters" in out.columns:
        u = pd.to_numeric(out["coordinateUncertaintyInMeters"], errors="coerce")
        out = out[u.isna() | (u <= max_uncertainty_m)]  # unknown uncertainty is kept
        steps.append((f"uncertainty <= {max_uncertainty_m:g} m", len(out)))

    if year_min is not None and "year" in out.columns:
        y = pd.to_numeric(out["year"], errors="coerce")
        out = out[y.isna() | (y >= year_min)]
        steps.append((f"year >= {year_min}", len(out)))

    out = out.assign(_lon5=out["lon"].round(5), _lat5=out["lat"].round(5))
    out = out.drop_duplicates(subset=["_lon5", "_lat5"]).drop(columns=["_lon5", "_lat5"])
    steps.append(("duplicate coordinates removed", len(out)))

    if thin_to_grid:
        row, col, _ = grid.index_of(out["lon"].to_numpy(), out["lat"].to_numpy())
        out = out.assign(_cell=row.astype(np.int64) * grid.width + col)
        out = out.drop_duplicates(subset="_cell").drop(columns="_cell")
        steps.append(("one record per grid cell", len(out)))

    return out.reset_index(drop=True), steps


# --------------------------------------------------------------------------- #
# Satellite imagery
# --------------------------------------------------------------------------- #
def cloud_mask_from_scl(scl: np.ndarray, invalid: tuple[int, ...] = SCL_INVALID) -> np.ndarray:
    """True where a pixel is cloud / shadow / no-data according to the SCL band."""
    return np.isin(scl, invalid)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red); NaN where the denominator is not positive."""
    red = red.astype("float32")
    nir = nir.astype("float32")
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / denom
    ndvi[~(denom > 0)] = np.nan
    return np.clip(ndvi, -1.0, 1.0)


def ndvi_composite(
    scenes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Cloud-masked median NDVI composite.

    ``scenes`` is a list of ``(red, nir, scl)`` arrays. Returns the composite and
    the per-pixel count of clear observations that fed it.
    """
    if not scenes:
        raise ValueError("no scenes supplied")
    stack = []
    for red, nir, scl in scenes:
        ndvi = compute_ndvi(red, nir)
        ndvi[cloud_mask_from_scl(scl)] = np.nan
        stack.append(ndvi)
    cube = np.stack(stack)
    count = np.sum(~np.isnan(cube), axis=0).astype("int16")
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            composite = np.nanmedian(cube, axis=0)
    return composite.astype("float32"), count


def fill_gaps(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill NaN pixels with the nearest valid value (e.g. pixels cloudy in every scene).

    Returns the filled array and the number of pixels filled. Appropriate for small
    gaps in a smooth field such as NDVI; collect more scenes if the gaps are large.
    """
    from scipy.ndimage import distance_transform_edt

    gaps = np.isnan(arr)
    n = int(gaps.sum())
    if n == 0 or n == arr.size:
        return arr, 0
    idx = distance_transform_edt(gaps, return_distances=False, return_indices=True)
    return arr[tuple(idx)].astype(arr.dtype), n


# --------------------------------------------------------------------------- #
# Clip + resample to the common grid
# --------------------------------------------------------------------------- #
def align_to_grid(path, grid: Grid, resampling: str = "bilinear", band: int = 1) -> np.ndarray:
    """Clip a raster to ``grid`` and resample it to the grid's resolution.

    Use ``bilinear``/``cubic`` for continuous data being upsampled, ``average``
    when aggregating a finer raster, and ``nearest`` for categorical bands.
    """
    method = Resampling[resampling]
    dst = np.full(grid.shape, np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=method,
        )
    return dst


def derive_slope(dem: np.ndarray, grid: Grid) -> np.ndarray:
    """Slope in degrees from a DEM on a geographic grid (Horn-style central differences)."""
    _, lat = grid.cell_centres()
    dy = grid.resolution * 111_320.0
    dx = np.maximum(dy * np.cos(np.radians(lat)), 1.0)
    filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dz_dy, dz_dx = np.gradient(filled, dy, axis=0), np.gradient(filled, axis=1) / dx
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    slope[np.isnan(dem)] = np.nan
    return slope.astype("float32")
