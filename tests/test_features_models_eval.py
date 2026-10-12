import numpy as np
import pandas as pd

from hsm.evaluation import boyce_index, evaluate, tss_threshold
from hsm.features import build_dataset, extract_at_points, prune_correlated
from hsm.grid import Grid
from hsm.integration import LayerStack
from hsm.mapping import area_by_class, classify
from hsm.modeling import make_models, split_train_test


def _stack():
    g = Grid(0, 40, 4, 44, 0.1)
    lon, lat = g.cell_centres()
    return LayerStack.from_layers({"x": lon.astype("float32"), "y": lat.astype("float32")}, g)


def test_extract_matches_cell_centre_values():
    s = _stack()
    df = extract_at_points(s, [0.55], [43.45])
    assert np.isclose(df["x"][0], 0.55, atol=0.06)
    assert np.isclose(df["y"][0], 43.45, atol=0.06)


def test_extract_outside_grid_is_nan():
    df = extract_at_points(_stack(), [50.0], [10.0])
    assert df.isna().all().all()


def test_build_dataset_labels_and_counts():
    s = _stack()
    occ = pd.DataFrame({"lon": [1.0, 2.0, 3.0], "lat": [41.0, 42.0, 43.0]})
    d = build_dataset(s, occ, n_background=50, seed=1)
    assert (d.label == 1).sum() == 3 and (d.label == 0).sum() == 50
    assert list(d.columns) == ["label", "lon", "lat", "x", "y"]
    assert not d.isna().any().any()


def test_build_dataset_is_reproducible():
    s = _stack()
    occ = pd.DataFrame({"lon": [1.0], "lat": [41.0]})
    a = build_dataset(s, occ, 30, seed=5)
    b = build_dataset(s, occ, 30, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_prune_correlated_keeps_priority_order():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    df = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(0, 0.05, 500), "c": rng.normal(size=500)})
    kept, dropped = prune_correlated(df, ["a", "b", "c"], 0.85)
    assert kept == ["a", "c"]
    assert dropped[0][0] == "b" and dropped[0][1] == "a"


def test_metrics_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    perfect = evaluate(y, y.astype(float), 0.5)
    assert perfect["auc"] == 1 and perfect["tss"] == 1 and perfect["kappa"] == 1
    rng = np.random.default_rng(0)
    rand = evaluate(y, rng.random(100), 0.5)
    assert 0.3 < rand["auc"] < 0.7


def test_tss_threshold_separates_classes():
    y = np.array([0] * 100 + [1] * 100)
    p = np.concatenate([np.linspace(0, 0.4, 100), np.linspace(0.6, 1, 100)])
    t = tss_threshold(y, p)
    assert 0.35 < t < 0.65


def test_boyce_positive_for_informative_negative_for_inverted():
    rng = np.random.default_rng(0)
    allp = rng.random(20000)
    pres_good = np.clip(rng.beta(5, 1.5, 300), 0, 1)  # skewed to high suitability
    assert boyce_index(pres_good, allp) > 0.5
    assert boyce_index(1 - pres_good, allp) < 0


def test_classify_and_area():
    s = np.array([[0.1, 0.3], [0.6, np.nan]])
    c = classify(s)
    assert c.tolist() == [[1, 2], [3, 0]]
    g = Grid(0, 40, 0.2, 40.2, 0.1)
    a = area_by_class(classify(np.array([[0.9, 0.1], [0.9, 0.1]])), g)
    assert abs(a["percent"].sum() - 100) < 0.1


def test_models_fit_and_split_is_stratified():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=600) > 1.0).astype(int)
    df = pd.DataFrame(X, columns=list("abc")).assign(label=y, lon=0.0, lat=0.0)
    tr, te = split_train_test(df, 0.3, 1)
    assert abs(tr.label.mean() - te.label.mean()) < 0.02
    cfg = {"random_forest": {"n_estimators": 50}, "algorithms": ["random_forest", "maxent_style"]}
    for name, m in make_models(cfg, 1).items():
        m.fit(tr[list("abc")].to_numpy(), tr.label.to_numpy())
        p = m.predict_proba(te[list("abc")].to_numpy())[:, 1]
        assert ((p >= 0) & (p <= 1)).all(), name
        assert evaluate(te.label.to_numpy(), p, 0.5)["auc"] > 0.8, name
