"""Stage 3 - data integration.

All environmental layers are held in one :class:`LayerStack`: a ``(layers, rows,
cols)`` cube on a single grid, with a shared validity mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid import Grid
from .io import read_raster, write_raster


@dataclass
class LayerStack:
    names: list[str]
    data: np.ndarray  # (n_layers, H, W), float32, NaN = no data
    grid: Grid

    def __post_init__(self):
        if self.data.shape[0] != len(self.names):
            raise ValueError("number of names must match number of layers")
        if self.data.shape[1:] != self.grid.shape:
            raise ValueError("layer shape does not match grid")

    @classmethod
    def from_layers(cls, layers: dict[str, np.ndarray], grid: Grid) -> "LayerStack":
        names = list(layers)
        data = np.stack([layers[n].astype("float32") for n in names])
        return cls(names, data, grid)

    @property
    def valid_mask(self) -> np.ndarray:
        """True where *every* layer has data."""
        return ~np.isnan(self.data).any(axis=0)

    def subset(self, names: list[str]) -> "LayerStack":
        idx = [self.names.index(n) for n in names]
        return LayerStack(list(names), self.data[idx], self.grid)

    def summary(self):
        import pandas as pd

        rows = []
        for n, band in zip(self.names, self.data):
            rows.append(
                {
                    "layer": n,
                    "min": float(np.nanmin(band)),
                    "mean": float(np.nanmean(band)),
                    "max": float(np.nanmax(band)),
                    "pct_nodata": float(100 * np.isnan(band).mean()),
                }
            )
        return pd.DataFrame(rows)

    def save(self, path) -> Path:
        return write_raster(path, self.data, self.grid, descriptions=self.names)

    @classmethod
    def load(cls, path) -> "LayerStack":
        arr, grid, names = read_raster(path)
        return cls([n or f"band{i+1}" for i, n in enumerate(names)], arr, grid)
