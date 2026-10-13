"""A tiny north-up, square-pixel, geographic grid definition.

Every raster in the pipeline is aligned to one :class:`Grid`, which is the
"common spatial framework" of stage 3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine, from_origin

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Grid:
    west: float
    south: float
    east: float
    north: float
    resolution: float
    crs: str = "EPSG:4326"

    @classmethod
    def from_config(cls, cfg: dict) -> "Grid":
        return cls(
            west=float(cfg["west"]),
            south=float(cfg["south"]),
            east=float(cfg["east"]),
            north=float(cfg["north"]),
            resolution=float(cfg["resolution"]),
            crs=str(cfg.get("crs", "EPSG:4326")),
        )

    @property
    def width(self) -> int:
        return int(round((self.east - self.west) / self.resolution))

    @property
    def height(self) -> int:
        return int(round((self.north - self.south) / self.resolution))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def transform(self) -> Affine:
        return from_origin(self.west, self.north, self.resolution, self.resolution)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(west, south, east, north)."""
        return (self.west, self.south, self.east, self.north)

    def buffered(self, degrees: float) -> "Grid":
        return Grid(
            self.west - degrees,
            self.south - degrees,
            self.east + degrees,
            self.north + degrees,
            self.resolution,
            self.crs,
        )

    def with_resolution(self, resolution: float) -> "Grid":
        return Grid(self.west, self.south, self.east, self.north, resolution, self.crs)

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lon, lat) 2-D arrays of cell-centre coordinates."""
        lon = self.west + (np.arange(self.width) + 0.5) * self.resolution
        lat = self.north - (np.arange(self.height) + 0.5) * self.resolution
        return np.meshgrid(lon, lat)

    def cell_area_km2(self) -> np.ndarray:
        """Spherical cell areas (km2), shape == grid shape."""
        lat_top = self.north - np.arange(self.height) * self.resolution
        lat_bot = lat_top - self.resolution
        dlon = np.radians(self.resolution)
        per_row = EARTH_RADIUS_KM**2 * dlon * (
            np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bot))
        )
        return np.repeat(per_row[:, None], self.width, axis=1)

    def index_of(self, lon, lat):
        """Map coordinates to (row, col, inside_mask)."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        col = np.floor((lon - self.west) / self.resolution).astype(int)
        row = np.floor((self.north - lat) / self.resolution).astype(int)
        inside = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        return row, col, inside

    def contains(self, lon, lat) -> np.ndarray:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return (
            (lon >= self.west) & (lon < self.east) & (lat > self.south) & (lat <= self.north)
        )
