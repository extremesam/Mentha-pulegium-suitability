"""Stage 8 - transfer: project a fitted model onto a region it was not trained on.

Projecting into a new region (here: a species' known range -> Nigeria) is an
*extrapolation* problem. Three safeguards are built in:

* **MESS** (multivariate environmental similarity surface, Elith et al. 2010):
  for every target cell, how similar are its predictor values to the training
  data? Negative = at least one predictor is outside the training range, i.e. the
  model is guessing. The *most dissimilar variable* (MoD) says which one.
* **Clamping**: predictor values are clipped to the training min/max before
  prediction, so a quadratic term cannot run away outside the range.
* **Model agreement**: the standard deviation between algorithms. Models tend to
  agree where data is plentiful and diverge when extrapolating.

The result is a *transfer suitability* map. It shows where conditions resemble
those where the species is known to live. It is not a forecast that the species
would establish there: dispersal, biotic interactions and land use are ignored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.features import geometry_mask

from .grid import Grid
from .integration import LayerStack


# --------------------------------------------------------------------------- #
# MESS
# --------------------------------------------------------------------------- #
def compute_mess(reference: pd.DataFrame, stack: LayerStack) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mess, mod)`` for every cell of ``stack``.

    ``reference`` holds the training values (one column per predictor; use the
    presence + background rows of the modelling table). ``mess`` is in percent;
    cells missing any predictor are NaN. ``mod`` is the index (into
    ``stack.names``) of the most dissimilar variable, -1 where undefined.
    """
    names = stack.names
    h, w = stack.grid.shape
    sims = np.full((len(names), h, w), np.nan, dtype="float32")
    for i, name in enumerate(names):
        ref = np.sort(reference[name].to_numpy(dtype="float64"))
        ref = ref[~np.isnan(ref)]
        lo, hi = ref[0], ref[-1]
        rng = hi - lo
        if rng <= 0:
            raise ValueError(f"predictor {name!r} is constant in the training data")
        p = stack.data[i].astype("float64")
        ok = ~np.isnan(p)
        f = np.full(p.shape, np.nan)
        f[ok] = 100.0 * np.searchsorted(ref, p[ok], side="left") / len(ref)  # % of reference < p
        s = np.where(
            f == 0,
            100.0 * (p - lo) / rng,
            np.where(f == 100, 100.0 * (hi - p) / rng, np.where(f <= 50, 2 * f, 2 * (100 - f))),
        )
        s[~ok] = np.nan
        sims[i] = s
    valid = ~np.isnan(sims).any(axis=0)
    mess = np.full((h, w), np.nan, dtype="float32")
    mod = np.full((h, w), -1, dtype="int16")
    mess[valid] = sims[:, valid].min(axis=0)
    mod[valid] = sims[:, valid].argmin(axis=0)
    return mess, mod


def clamp_stack(stack: LayerStack, reference: pd.DataFrame) -> LayerStack:
    """Clip every layer to the training min/max (Maxent-style clamping)."""
    out = stack.data.copy()
    for i, name in enumerate(stack.names):
        out[i] = np.clip(out[i], reference[name].min(), reference[name].max())
    return LayerStack(list(stack.names), out, stack.grid)


def variable_range_report(reference: pd.DataFrame, stack: LayerStack) -> pd.DataFrame:
    """Per predictor: training range vs target range and share of target cells outside training range."""
    rows = []
    for i, name in enumerate(stack.names):
        band = stack.data[i]
        ok = ~np.isnan(band)
        lo, hi = float(reference[name].min()), float(reference[name].max())
        below = float(100 * (band[ok] < lo).mean())
        above = float(100 * (band[ok] > hi).mean())
        rows.append(
            {
                "variable": name,
                "train_min": lo,
                "train_max": hi,
                "target_min": float(np.nanmin(band)),
                "target_max": float(np.nanmax(band)),
                "pct_target_below_train": round(below, 2),
                "pct_target_above_train": round(above, 2),
                "pct_target_outside_train": round(below + above, 2),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Country / region mask
# --------------------------------------------------------------------------- #
def mask_from_geojson(path, grid: Grid) -> np.ndarray:
    """Boolean array, True for cells whose centre lies inside any polygon in a GeoJSON file."""
    with open(path) as fh:
        gj = json.load(fh)
    t = gj.get("type")
    if t == "FeatureCollection":
        geoms = [f["geometry"] for f in gj["features"] if f.get("geometry")]
    elif t == "Feature":
        geoms = [gj["geometry"]]
    else:
        geoms = [gj]
    if not geoms:
        raise ValueError(f"no geometries found in {path}")
    outside = geometry_mask(geoms, out_shape=grid.shape, transform=grid.transform, invert=False, all_touched=False)
    return ~outside


# --------------------------------------------------------------------------- #
# Transfer prediction
# --------------------------------------------------------------------------- #
def project_models(
    fitted: dict,
    members: list[str],
    stack: LayerStack,
    features: list[str],
    chunk: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Ensemble mean, between-model SD, and each model's surface on a (clamped) stack."""
    from .modeling import predict_surface

    surfaces = {
        n: predict_surface(lambda X, m=fitted[n]: m.predict_proba(X)[:, 1], stack, features, chunk)
        for n in members
    }
    cube = np.stack([surfaces[m] for m in members])
    return np.nanmean(cube, axis=0).astype("float32"), np.nanstd(cube, axis=0).astype("float32"), surfaces


def transfer_area_table(
    classes: np.ndarray, in_domain: np.ndarray, grid: Grid, class_names: dict[int, str]
) -> pd.DataFrame:
    """Area per suitability class, split into inside / outside the model's training domain (MESS >= 0)."""
    area = grid.cell_area_km2()
    valid = classes != 0
    total = float(area[valid].sum())
    rows = []
    for k in sorted(class_names, reverse=True):
        sel = classes == k
        tot = float(area[sel].sum())
        dom = float(area[sel & in_domain].sum())
        rows.append(
            {
                "class": class_names[k],
                "area_km2_total": round(tot, 1),
                "percent_of_region": round(100 * tot / total, 2),
                "area_km2_in_domain": round(dom, 1),
                "percent_of_region_in_domain": round(100 * dom / total, 2),
            }
        )
    return pd.DataFrame(rows)
