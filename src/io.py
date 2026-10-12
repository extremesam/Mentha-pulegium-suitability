"""GeoTIFF read/write helpers with NaN <-> nodata handling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .grid import Grid


def write_raster(
    path,
    array: np.ndarray,
    grid: Grid,
    nodata: float | int | None = -9999.0,
    dtype: str = "float32",
    descriptions: list[str] | None = None,
) -> Path:
    """Write a 2-D (H, W) or 3-D (bands, H, W) array as a compressed GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[1:] != grid.shape:
        raise ValueError(f"array shape {arr.shape[1:]} does not match grid {grid.shape}")
    if np.issubdtype(np.dtype(dtype), np.floating) and nodata is not None:
        arr = np.where(np.isnan(arr), nodata, arr)
    arr = arr.astype(dtype)
    profile = dict(
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=arr.shape[0],
        dtype=dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        if descriptions:
            for i, desc in enumerate(descriptions, start=1):
                dst.set_band_description(i, desc)
    return path


def read_raster(path) -> tuple[np.ndarray, Grid, list[str | None]]:
    """Read all bands as float32 (nodata -> NaN). Returns (array, grid, band names)."""
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        if src.nodata is not None:
            arr[arr == np.float32(src.nodata)] = np.nan
        t = src.transform
        grid = Grid(
            west=t.c,
            south=t.f + src.height * t.e,
            east=t.c + src.width * t.a,
            north=t.f,
            resolution=abs(t.a),
            crs=src.crs.to_string() if src.crs else "EPSG:4326",
        )
        names = list(src.descriptions)
    return arr, grid, names
