"""Stage 5 - modelling.

Three presence/background learners plus a weighted-free mean ensemble:

* ``random_forest``  - sklearn RandomForest (balanced class weights)
* ``maxent_style``   - L1-regularised logistic regression on linear + quadratic
  features. Maxent is equivalent to a regularised Poisson point-process model
  (Renner & Warton 2013), so this is a close, dependency-free analogue. It is
  *not* the Maxent Java program and is named accordingly.
* ``boosted_trees``  - sklearn HistGradientBoosting

Outputs are *relative suitability indices* in [0, 1]. Because absences are
pseudo-absences, they are not calibrated occurrence probabilities.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .integration import LayerStack

log = logging.getLogger(__name__)


def _add_quadratic(X):
    X = np.asarray(X, dtype=float)
    return np.hstack([X, X**2])


def _l1_logistic_cv(seed: int) -> LogisticRegressionCV:
    """L1-penalised logistic regression with CV over C, across scikit-learn versions.

    scikit-learn >= 1.8 replaces ``penalty="l1"`` with ``l1_ratios=(1,)`` and warns
    about legacy fitted attributes; older versions only know ``penalty``.
    """
    common = dict(Cs=10, cv=5, scoring="roc_auc", class_weight="balanced", max_iter=2000, random_state=seed)
    try:
        return LogisticRegressionCV(l1_ratios=(1,), solver="saga", use_legacy_attributes=False, **common)
    except TypeError:
        return LogisticRegressionCV(penalty="l1", solver="liblinear", **common)


def make_models(cfg: dict, seed: int) -> dict[str, object]:
    """Instantiate the configured models."""
    rf = cfg.get("random_forest", {})
    gb = cfg.get("boosted_trees", {})
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=rf.get("n_estimators", 500),
            min_samples_leaf=rf.get("min_samples_leaf", 5),
            max_features=rf.get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "maxent_style": Pipeline(
            [
                ("scale", StandardScaler()),
                ("features", FunctionTransformer(_add_quadratic)),
                ("logit", _l1_logistic_cv(seed)),
            ]
        ),
        "boosted_trees": HistGradientBoostingClassifier(
            learning_rate=gb.get("learning_rate", 0.05),
            max_iter=gb.get("max_iter", 300),
            max_leaf_nodes=gb.get("max_leaf_nodes", 15),
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        ),
    }
    enabled = cfg.get("algorithms")
    if enabled:
        models = {k: v for k, v in models.items() if k in enabled}
    return models


def split_train_test(df: pd.DataFrame, test_size: float, seed: int):
    """Stratified random split (default 70 % train / 30 % test)."""
    return train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)


def fit_models(models: dict, train: pd.DataFrame, features: list[str]) -> dict:
    fitted = {}
    for name, model in models.items():
        m = clone(model)
        m.fit(train[features].to_numpy(), train["label"].to_numpy())
        fitted[name] = m
        log.info("fitted %s", name)
    return fitted


def predict_df(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(df[features].to_numpy())[:, 1]


def ensemble_predict(fitted: dict, df: pd.DataFrame, features: list[str], members: list[str]) -> np.ndarray:
    return np.mean([predict_df(fitted[m], df, features) for m in members], axis=0)


def predict_surface(
    predictor, stack: LayerStack, features: list[str], chunk: int = 200_000
) -> np.ndarray:
    """Apply ``predictor(X) -> probs`` to every valid cell; NaN elsewhere."""
    sub = stack.subset(features)
    valid = sub.valid_mask
    flat = sub.data.reshape(len(features), -1).T
    idx = np.flatnonzero(valid.ravel())
    out = np.full(flat.shape[0], np.nan, dtype="float32")
    for start in range(0, len(idx), chunk):
        sel = idx[start : start + chunk]
        out[sel] = predictor(flat[sel])
    return out.reshape(stack.grid.shape)


def spatial_block_cv(
    model, df: pd.DataFrame, features: list[str], block_deg: float, n_splits: int = 5
) -> list[float]:
    """AUC across geographic blocks (whole blocks held out) - a harder test than a random split."""
    blocks = (
        (np.floor(df["lon"] / block_deg).astype(int).astype(str))
        + "_"
        + (np.floor(df["lat"] / block_deg).astype(int).astype(str))
    )
    groups = pd.factorize(blocks)[0]
    n_splits = min(n_splits, len(np.unique(groups)))
    aucs = []
    X, y = df[features].to_numpy(), df["label"].to_numpy()
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        m = clone(model).fit(X[tr], y[tr])
        aucs.append(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
    return aucs


def variable_importance(
    model, test: pd.DataFrame, features: list[str], seed: int, n_repeats: int = 10
) -> pd.DataFrame:
    """Permutation importance (drop in test AUC when a predictor is shuffled)."""
    res = permutation_importance(
        model,
        test[features].to_numpy(),
        test["label"].to_numpy(),
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    imp = pd.DataFrame(
        {"variable": features, "importance": res.importances_mean, "std": res.importances_std}
    )
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def response_curves(
    predictor, df: pd.DataFrame, features: list[str], n_points: int = 40, n_rows: int = 500, seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Partial-dependence style curves: vary one predictor, average over a background sample."""
    rng = np.random.default_rng(seed)
    base = df[features].sample(min(n_rows, len(df)), random_state=seed).to_numpy()
    curves = {}
    for j, f in enumerate(features):
        lo, hi = np.percentile(df[f], [2, 98])
        grid = np.linspace(lo, hi, n_points)
        y = []
        for v in grid:
            X = base.copy()
            X[:, j] = v
            y.append(float(np.mean(predictor(X))))
        curves[f] = pd.DataFrame({"value": grid, "suitability": y})
    return curves
