"""Transfer (projection) stage: MESS, clamping, masks, background buffer, data-prep script, smoke run."""
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from hsm.features import buffer_mask
from hsm.grid import Grid
from hsm.integration import LayerStack
from hsm.pipeline import load_config, run
from hsm.projection import (
    clamp_stack,
    compute_mess,
    mask_from_geojson,
    project_models,
    transfer_area_table,
    variable_range_report,
)

REPO = Path(__file__).resolve().parents[1]


def _stack_1d(values_by_name: dict[str, list[float]]) -> LayerStack:
    n = len(next(iter(values_by_name.values())))
    g = Grid(0, 0, n * 0.1, 0.1, 0.1)
    return LayerStack.from_layers({k: np.array([v], dtype="float32") for k, v in values_by_name.items()}, g)


def test_mess_single_variable_known_values():
    ref = pd.DataFrame({"a": np.linspace(0, 100, 1001)})
    s = _stack_1d({"a": [50, 25, 0, -10, 110]})
    mess, _ = compute_mess(ref, s)
    centre, quarter, at_min, below, above = mess[0]
    assert centre == pytest.approx(100, abs=1)       # median of the training data = most similar
    assert quarter == pytest.approx(50, abs=1)
    assert at_min == pytest.approx(0, abs=0.1)
    assert below == pytest.approx(-10, abs=0.1)      # 10% of the range below the minimum
    assert above == pytest.approx(-10, abs=0.1)      # 10% of the range above the maximum


def test_mess_takes_minimum_and_reports_most_dissimilar_variable():
    ref = pd.DataFrame({"a": np.linspace(0, 100, 1001), "b": np.linspace(0, 10, 1001)})
    s = _stack_1d({"a": [50, 50], "b": [5, 20]})  # second cell: b is far outside
    mess, mod = compute_mess(ref, s)
    assert mess[0, 0] > 90 and mess[0, 1] < 0
    assert mod[0, 1] == 1  # 'b' is the culprit


def test_mess_nan_where_any_layer_missing_and_constant_predictor_rejected():
    ref = pd.DataFrame({"a": np.linspace(0, 1, 50), "b": np.linspace(0, 1, 50)})
    g = Grid(0, 0, 0.2, 0.1, 0.1)
    st = LayerStack.from_layers({"a": np.array([[0.5, np.nan]]), "b": np.array([[0.5, 0.5]])}, g)
    mess, mod = compute_mess(ref, st)
    assert not np.isnan(mess[0, 0]) and np.isnan(mess[0, 1]) and mod[0, 1] == -1
    with pytest.raises(ValueError):
        compute_mess(pd.DataFrame({"a": np.ones(10), "b": np.ones(10)}), st)


def test_clamp_and_range_report():
    ref = pd.DataFrame({"a": [0.0, 10.0]})
    s = _stack_1d({"a": [-5, 5, 20, 30]})
    c = clamp_stack(s, ref)
    assert list(c.data[0, 0]) == [0, 5, 10, 10]
    rep = variable_range_report(ref, s).iloc[0]
    assert rep["pct_target_below_train"] == 25 and rep["pct_target_above_train"] == 50
    assert rep["pct_target_outside_train"] == 75


def test_mask_from_geojson(tmp_path):
    g = Grid(0, 0, 4, 4, 1.0)
    poly = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 4], [0, 4], [0, 0]]]}  # western half
    for wrapper in (
        poly,
        {"type": "Feature", "properties": {}, "geometry": poly},
        {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": poly}]},
    ):
        p = tmp_path / "m.geojson"
        p.write_text(json.dumps(wrapper))
        m = mask_from_geojson(p, g)
        assert m.shape == g.shape
        assert m[:, :2].all() and not m[:, 2:].any()


def test_buffer_mask_limits_to_distance():
    g = Grid(0, 0, 10, 10, 0.1)
    s = LayerStack.from_layers({"x": np.ones(g.shape, dtype="float32")}, g)
    m = buffer_mask(s, [5.0], [5.0], buffer_km=222.64)  # = 2.0 degrees
    lon, lat = g.cell_centres()
    dist = np.hypot(lon - 5.05, lat - 4.95)
    assert m[dist < 1.8].all()          # clearly inside
    assert not m[dist > 2.3].any()      # clearly outside
    assert 0 < m.mean() < 0.2


def test_project_models_mean_sd_and_area_table():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] > 0).astype(int)
    fitted = {
        "m1": RandomForestClassifier(30, random_state=0).fit(X, y),
        "m2": RandomForestClassifier(30, random_state=1, max_depth=2).fit(X, y),
    }
    g = Grid(0, 0, 3, 3, 1.0)
    st = LayerStack.from_layers(
        {"a": rng.normal(size=g.shape).astype("float32"), "b": rng.normal(size=g.shape).astype("float32")}, g
    )
    mean, sd, per = project_models(fitted, ["m1", "m2"], st, ["a", "b"])
    assert mean.shape == g.shape and ((mean >= 0) & (mean <= 1)).all() and (sd >= 0).all()
    assert set(per) == {"m1", "m2"}

    classes = np.array([[4, 3, 2], [1, 1, 0], [4, 4, 4]], dtype="uint8")
    dom = np.array([[True, True, False], [False, True, True], [True, True, True]])
    tbl = transfer_area_table(classes, dom, g, {4: "High", 3: "Moderate", 2: "Low", 1: "Very low"})
    assert tbl["percent_of_region"].sum() == pytest.approx(100, abs=0.1)
    assert (tbl["area_km2_in_domain"] <= tbl["area_km2_total"]).all()


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location("prepare_real_data", REPO / "scripts" / "prepare_real_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prepare_script_extracts_and_renames(tmp_path):
    prep = _load_prepare_script()
    bio_zip, elev_zip = tmp_path / "bio.zip", tmp_path / "elev.zip"
    with zipfile.ZipFile(bio_zip, "w") as zf:
        for i in range(1, 20):
            zf.writestr(f"wc2.1_5m_bio_{i}.tif", f"bio{i}".encode())
    with zipfile.ZipFile(elev_zip, "w") as zf:
        zf.writestr("wc2.1_5m_elev.tif", b"elev")
    out = tmp_path / "layers"
    written = prep.extract_worldclim(bio_zip, elev_zip, out)
    names = sorted(p.name for p in written)
    assert names == sorted(["bio1.tif", "bio5.tif", "bio6.tif", "bio12.tif", "bio15.tif", "bio17.tif", "elevation_dem.tif"])
    assert (out / "bio1.tif").read_bytes() == b"bio1"      # bio1 is not confused with bio10..bio19
    assert (out / "bio12.tif").read_bytes() == b"bio12"
    assert (out / "elevation_dem.tif").read_bytes() == b"elev"


def test_prepare_script_reports_missing_layer(tmp_path):
    prep = _load_prepare_script()
    bad = tmp_path / "bio.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("wc2.1_5m_bio_1.tif", b"x")
    el = tmp_path / "e.zip"
    with zipfile.ZipFile(el, "w") as zf:
        zf.writestr("wc2.1_5m_elev.tif", b"x")
    with pytest.raises(FileNotFoundError):
        prep.extract_worldclim(bad, el, tmp_path / "o")


def _small(cfg):
    cfg["study_area"]["resolution"] = 0.2
    cfg["background"]["n_points"] = 800
    cfg["modeling"]["random_forest"]["n_estimators"] = 60
    cfg["modeling"]["boosted_trees"]["max_iter"] = 60
    cfg["modeling"]["spatial_cv"]["enabled"] = False
    return cfg


def test_transfer_pipeline_smoke(tmp_path):
    cfg = _small(load_config("config/transfer_demo.yaml"))
    cfg["projection"]["grid"]["resolution"] = 0.2
    summary = run(cfg, mode="demo", root=tmp_path)
    proj = summary["projection"]
    for rel in (
        "results_transfer_demo/rasters/transfer_suitability_index.tif",
        "results_transfer_demo/rasters/transfer_mess.tif",
        "results_transfer_demo/rasters/transfer_model_sd.tif",
        "results_transfer_demo/figures/09_transfer_suitability.png",
        "results_transfer_demo/figures/10_transfer_extrapolation_diagnostics.png",
        "results_transfer_demo/transfer_class_area.csv",
        "results_transfer_demo/transfer_variable_ranges.csv",
    ):
        assert (tmp_path / rel).exists(), rel
    assert 0 < proj["pct_cells_in_training_domain"] < 100  # the demo target is partly novel by design
    assert abs(sum(c["percent_of_region"] for c in proj["class_area"]) - 100) < 0.2
    # the target is warmer by construction, so bio1 must be the variable most often out of range
    worst = max(proj["variable_ranges"], key=lambda r: r["pct_target_outside_train"])
    assert worst["variable"] == "bio1"
    assert proj["demo_validation"]["spearman_all"] > 0.4


def test_transfer_rejects_scene_derived_predictors(tmp_path):
    cfg = _small(load_config("config/config.yaml"))  # has the NDVI layer
    cfg["projection"] = {"enabled": True, "name": "x", "grid": {"west": 20, "south": 5, "east": 30, "north": 13, "resolution": 0.2}}
    with pytest.raises(ValueError, match="satellite scenes"):
        run(cfg, mode="demo", root=tmp_path)
