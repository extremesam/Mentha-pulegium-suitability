"""Stage 4 - feature extraction.

Turns (layer stack + occurrence points) into a tabular modelling dataset of
presences and background pseudo-absences.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .integration import LayerStack

log = logging.getLogger(__name__)


def extract_at_points(stack: LayerStack, lon, lat) -> pd.DataFrame:
    """Sample every layer at the given coordinates (NaN outside the grid)."""
    row, col, inside = stack.grid.index_of(lon, lat)
    out = np.full((len(row), len(stack.names)), np.nan, dtype="float32")
    out[inside] = stack.data[:, row[inside], col[inside]].T
    return pd.DataFrame(out, columns=stack.names)


def buffer_mask(stack: LayerStack, lon, lat, buffer_km: float) -> np.ndarray:
    """True for cells within ``buffer_km`` of any presence cell.

    Distance is measured in cell units and converted with 111.32 km per degree, so it
    is approximate (it overstates east-west distances away from the equator). Use it to
    restrict background sampling to the area the species could plausibly have reached
    rather than the whole study area.
    """
    from scipy.ndimage import distance_transform_edt

    row, col, inside = stack.grid.index_of(lon, lat)
    seeds = np.zeros(stack.grid.shape, dtype=bool)
    seeds[row[inside], col[inside]] = True
    dist_cells = distance_transform_edt(~seeds)
    return dist_cells * stack.grid.resolution * 111.32 <= buffer_km


def sample_background(
    stack: LayerStack,
    n: int,
    rng: np.random.Generator,
    exclude_cells: set[int] | None = None,
    allowed: np.ndarray | None = None,
) -> pd.DataFrame:
    """Draw ``n`` random background points from valid cells (jittered within the cell).

    ``allowed`` optionally restricts sampling to a boolean (H, W) mask.
    """
    valid = stack.valid_mask if allowed is None else (stack.valid_mask & allowed)
    rows, cols = np.nonzero(valid)
    if exclude_cells:
        flat = rows.astype(np.int64) * stack.grid.width + cols
        keep = ~np.isin(flat, list(exclude_cells))
        rows, cols = rows[keep], cols[keep]
    n = min(n, len(rows))
    pick = rng.choice(len(rows), size=n, replace=False)
    r, c = rows[pick], cols[pick]
    res = stack.grid.resolution
    lon = stack.grid.west + (c + rng.random(n)) * res
    lat = stack.grid.north - (r + rng.random(n)) * res
    return pd.DataFrame({"lon": lon, "lat": lat})


def build_dataset(
    stack: LayerStack,
    occurrences: pd.DataFrame,
    n_background: int,
    seed: int = 42,
    exclude_presence_cells: bool = False,
    buffer_km: float | None = None,
) -> pd.DataFrame:
    """Create the modelling table: ``label`` (1 presence / 0 background), lon, lat, layers.

    ``buffer_km`` restricts background points to within that distance of a presence cell.
    """
    rng = np.random.default_rng(seed)
    pres = occurrences[["lon", "lat"]].reset_index(drop=True)
    pres_env = extract_at_points(stack, pres["lon"], pres["lat"])
    presence = pd.concat([pres, pres_env], axis=1).assign(label=1)

    exclude = None
    if exclude_presence_cells:
        row, col, inside = stack.grid.index_of(pres["lon"], pres["lat"])
        exclude = set((row[inside].astype(np.int64) * stack.grid.width + col[inside]).tolist())

    allowed = buffer_mask(stack, pres["lon"], pres["lat"], buffer_km) if buffer_km else None
    bg = sample_background(stack, n_background, rng, exclude, allowed)
    bg_env = extract_at_points(stack, bg["lon"], bg["lat"])
    background = pd.concat([bg, bg_env], axis=1).assign(label=0)

    data = pd.concat([presence, background], ignore_index=True)
    before = len(data)
    data = data.dropna(subset=stack.names).reset_index(drop=True)
    if len(data) < before:
        log.warning("dropped %d rows with missing predictor values", before - len(data))
    return data[["label", "lon", "lat", *stack.names]]


def prune_correlated(
    df: pd.DataFrame, names: list[str], threshold: float = 0.85
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Greedy collinearity filter on |Pearson r|.

    Walks the predictors in the order given; a predictor is dropped if it
    correlates above ``threshold`` with one already kept. Put the variables you
    consider most ecologically meaningful first. Returns ``(kept, dropped)`` where
    ``dropped`` holds ``(dropped, kept_partner, r)``.
    """
    corr = df[names].corr().abs()
    kept: list[str] = []
    dropped: list[tuple[str, str, float]] = []
    for n in names:
        clash = [(k, corr.loc[n, k]) for k in kept if corr.loc[n, k] > threshold]
        if clash:
            partner, r = max(clash, key=lambda t: t[1])
            dropped.append((n, partner, float(r)))
        else:
            kept.append(n)
    return kept, dropped
