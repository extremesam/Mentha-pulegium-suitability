"""GBIF client is tested against a fake session (no network needed); pipeline gets a small smoke run."""
import json

import numpy as np
import pandas as pd

from hsm.data_collection import fetch_gbif_occurrences, load_occurrence_csv, match_taxon_key
from hsm.pipeline import load_config, run


class FakeResponse:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, records_per_page=2, total=5):
        self.calls = []
        self.total = total
        self.page = records_per_page

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url.endswith("/species/match"):
            return FakeResponse({"usageKey": 2927437})
        off = params["offset"]
        n = min(self.page, self.total - off, params["limit"])
        results = [
            {"gbifID": str(off + i), "species": "Mentha pulegium", "decimalLongitude": -3.0 + i, "decimalLatitude": 40.0, "year": 2015}
            for i in range(n)
        ]
        return FakeResponse({"results": results, "endOfRecords": off + n >= self.total})


def test_taxon_match():
    assert match_taxon_key("Mentha pulegium", FakeSession()) == 2927437


def test_gbif_pagination_and_params():
    s = FakeSession(records_per_page=2, total=5)
    df = fetch_gbif_occurrences("Mentha pulegium", (-10, 36, 4, 44), limit=100, page_size=2, session=s)
    assert len(df) == 5
    assert {"lon", "lat", "year"} <= set(df.columns)
    search = [p for u, p in s.calls if u.endswith("/occurrence/search")]
    assert [p["offset"] for p in search] == [0, 2, 4]
    assert search[0]["decimalLongitude"] == "-10,4" and search[0]["decimalLatitude"] == "36,44"
    assert search[0]["hasCoordinate"] == "true" and search[0]["occurrenceStatus"] == "PRESENT"


def test_gbif_respects_limit():
    s = FakeSession(records_per_page=300, total=1000)
    df = fetch_gbif_occurrences("x", (0, 0, 1, 1), limit=7, page_size=300, session=s)
    assert len(df) == 7


def test_csv_loader_renames_gbif_columns(tmp_path):
    p = tmp_path / "o.csv"
    pd.DataFrame({"decimalLongitude": [1.0], "decimalLatitude": [2.0]}).to_csv(p, index=False)
    assert list(load_occurrence_csv(p).columns) == ["lon", "lat"]


def test_pipeline_smoke_demo(tmp_path):
    cfg = load_config("config/config.yaml")
    cfg["study_area"]["resolution"] = 0.2  # coarse + fast
    cfg["background"]["n_points"] = 800
    cfg["modeling"]["random_forest"]["n_estimators"] = 60
    cfg["modeling"]["boosted_trees"]["max_iter"] = 60
    cfg["modeling"]["spatial_cv"]["enabled"] = False
    summary = run(cfg, mode="demo", root=tmp_path)
    for rel in (
        "results/metrics.json",
        "results/rasters/suitability_index.tif",
        "results/rasters/suitability_classes.tif",
        "results/figures/07_suitability_classes.png",
        "data/processed/environmental_stack.tif",
        "data/processed/modeling_dataset.csv",
    ):
        assert (tmp_path / rel).exists(), rel
    m = summary["metrics"]
    assert m["ensemble"]["auc"] > 0.7
    assert m["demo_validation"]["spearman_rho_vs_true_suitability"] > 0.5
    saved = json.loads((tmp_path / "results/metrics.json").read_text())
    assert saved["species"] == "Mentha pulegium"
    assert abs(sum(c["percent"] for c in saved["class_area"]) - 100) < 0.2
