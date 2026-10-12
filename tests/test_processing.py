import numpy as np
import pandas as pd

from hsm.grid import Grid
from hsm.io import write_raster
from hsm.processing import (
    align_to_grid,
    clean_occurrences,
    cloud_mask_from_scl,
    compute_ndvi,
    derive_slope,
    fill_gaps,
    ndvi_composite,
)


def test_ndvi_known_values():
    red = np.array([0.1, 0.2, 0.0])
    nir = np.array([0.5, 0.2, 0.0])
    ndvi = compute_ndvi(red, nir)
    assert np.isclose(ndvi[0], (0.5 - 0.1) / 0.6)
    assert ndvi[1] == 0
    assert np.isnan(ndvi[2])  # 0/0 -> NaN, not an exception


def test_cloud_mask_flags_scl_classes():
    scl = np.array([4, 5, 8, 9, 3, 10, 6])
    assert list(cloud_mask_from_scl(scl)) == [False, False, True, True, True, True, False]


def test_composite_ignores_cloudy_scene():
    clear = (np.full((2, 2), 0.1), np.full((2, 2), 0.5), np.full((2, 2), 4))
    cloudy = (np.full((2, 2), 0.5), np.full((2, 2), 0.5), np.full((2, 2), 9))  # NDVI 0 but masked
    comp, count = ndvi_composite([clear, clear, cloudy])
    assert np.allclose(comp, (0.5 - 0.1) / 0.6)
    assert (count == 2).all()


def test_composite_all_cloudy_is_nan():
    cloudy = (np.full((2, 2), 0.5), np.full((2, 2), 0.5), np.full((2, 2), 9))
    comp, count = ndvi_composite([cloudy])
    assert np.isnan(comp).all() and (count == 0).all()


def test_fill_gaps_nearest():
    a = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, np.nan]], dtype="float32")
    filled, n = fill_gaps(a)
    assert n == 2 and not np.isnan(filled).any()
    assert filled[0, 1] in (1.0, 2.0, 3.0)


def _occ():
    return pd.DataFrame(
        {
            "lon": [1.0, 1.01, np.nan, 0.0, 99.0, 3.5, 2.0, 2.0, 1.5],
            "lat": [40.0, 39.9, 40.0, 0.0, 40.0, 40.5, 41.0, 41.0, 41.5],
            "coordinateUncertaintyInMeters": [10, 10, 10, 10, 10, 50000, 10, 10, np.nan],
            "year": [2010, 2011, 2012, 2013, 2014, 2015, 1950, 2010, 2012],
        }
    )


def test_clean_occurrences_each_rule():
    g = Grid(0, 36, 4, 44, 0.5)
    out, steps = clean_occurrences(_occ(), g, max_uncertainty_m=5000, year_min=1990, thin_to_grid=True)
    names = [s for s, _ in steps]
    counts = dict(steps)
    assert counts["raw records"] == 9
    assert counts["has coordinates"] == 8
    assert counts["valid, non-(0,0) coordinates"] == 7  # the (0, 0) record is gone
    assert counts["inside study area"] == 6  # lon=99 is outside the grid
    assert "one record per grid cell" in names
    # 1.0/40.0 and 1.01/39.9 share a 0.5 deg cell -> thinned to one
    assert len(out) == 3
    assert out["lon"].notna().all()


def test_align_resamples_and_clips(tmp_path):
    src = Grid(-2, 34, 8, 46, 0.5)  # bigger extent, coarser
    arr = np.tile(np.linspace(0, 1, src.width, dtype="float32"), (src.height, 1))
    p = write_raster(tmp_path / "s.tif", arr, src)
    dst = Grid(0, 36, 4, 44, 0.25)
    out = align_to_grid(p, dst, "bilinear")
    assert out.shape == dst.shape
    assert not np.isnan(out).any()
    assert out[:, 0].mean() < out[:, -1].mean()  # gradient preserved after clipping


def test_slope_flat_is_zero_and_ramp_positive():
    g = Grid(0, 40, 2, 42, 0.1)
    flat = np.full(g.shape, 100.0, dtype="float32")
    assert np.nanmax(derive_slope(flat, g)) < 1e-4
    ramp = np.tile(np.linspace(0, 2000, g.width, dtype="float32"), (g.height, 1))
    # analytic check: 105.3 m rise per ~8.4 km cell at ~41N is atan(0.0125) = ~0.72 degrees
    assert 0.65 < np.nanmean(derive_slope(ramp, g)) < 0.8
