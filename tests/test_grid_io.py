import numpy as np

from hsm.grid import Grid
from hsm.io import read_raster, write_raster


def make_grid():
    return Grid(-10, 36, 4, 44, 0.5)


def test_shape_and_index_roundtrip():
    g = make_grid()
    assert g.shape == (16, 28)
    # NW corner cell centre maps to (0, 0); SE corner cell centre to the last cell
    row, col, inside = g.index_of([-9.75, 3.75], [43.75, 36.25])
    assert inside.all()
    assert list(row) == [0, 15] and list(col) == [0, 27]


def test_index_flags_outside_points():
    g = make_grid()
    _, _, inside = g.index_of([-11, 0, 10], [40, 50, 40])
    assert list(inside) == [False, False, False]


def test_cell_area_shrinks_poleward_and_totals_sensibly():
    g = make_grid()
    a = g.cell_area_km2()
    assert a[0, 0] < a[-1, 0]  # northern row is smaller
    # 14 deg x 8 deg at ~40N is roughly 1.2 million km2
    assert 1.0e6 < a.sum() < 1.4e6


def test_raster_roundtrip_preserves_nan_and_grid(tmp_path):
    g = make_grid()
    arr = np.random.default_rng(0).random(g.shape).astype("float32")
    arr[3, 4] = np.nan
    p = write_raster(tmp_path / "x.tif", arr, g, descriptions=["a"])
    back, g2, names = read_raster(p)
    assert np.isnan(back[0, 3, 4])
    np.testing.assert_allclose(back[0][~np.isnan(arr)], arr[~np.isnan(arr)], rtol=1e-6)
    assert (g2.west, g2.north, g2.resolution) == (g.west, g.north, g.resolution)
    assert names == ["a"]
