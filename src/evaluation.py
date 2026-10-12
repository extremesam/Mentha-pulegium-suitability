"""Stage 6 - model evaluation.

AUC, accuracy, Cohen's kappa, TSS (true skill statistic) and the continuous
Boyce index, all computed on the held-out test split.

Binary metrics need a threshold. It is chosen by maximising TSS on
*cross-validated training predictions* so the test set is never used to tune it.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_predict


def tss_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises sensitivity + specificity - 1."""
    best_t, best_tss = 0.5, -np.inf
    for t in np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 197))):
        pred = y_prob >= t
        tp = np.sum(pred & (y_true == 1))
        fn = np.sum(~pred & (y_true == 1))
        tn = np.sum(~pred & (y_true == 0))
        fp = np.sum(pred & (y_true == 0))
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if sens + spec - 1 > best_tss:
            best_tss, best_t = sens + spec - 1, float(t)
    return best_t


def cv_threshold(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    """TSS-optimal threshold from out-of-fold training predictions."""
    oof = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1]
    return tss_threshold(y, oof)


def boyce_index(
    pred_presence: np.ndarray, pred_all: np.ndarray, n_windows: int = 100, window: float = 0.1
) -> float:
    """Continuous Boyce index (Hirzel et al. 2006): Spearman rho of the predicted/expected
    presence ratio against suitability. +1 = perfect, 0 = random, <0 = counter-predictive."""
    pred_all = pred_all[~np.isnan(pred_all)]
    lo, hi = float(pred_all.min()), float(pred_all.max())
    width = (hi - lo) * window
    if width <= 0:
        return float("nan")
    centres = np.linspace(lo, hi - width, n_windows)
    ratios, used = [], []
    for c in centres:
        p = np.mean((pred_presence >= c) & (pred_presence < c + width))
        e = np.mean((pred_all >= c) & (pred_all < c + width))
        if e > 0:
            ratios.append(p / e)
            used.append(c)
    if len(used) < 5:
        return float("nan")
    return float(spearmanr(used, ratios).correlation)


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    pred_all: np.ndarray | None = None,
) -> dict:
    """Metric bundle for one model on the test split."""
    y_true = np.asarray(y_true)
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    out = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "kappa": float(cohen_kappa_score(y_true, pred)),
        "tss": float(sens + spec - 1),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "threshold": float(threshold),
    }
    if pred_all is not None:
        out["boyce"] = boyce_index(y_prob[y_true == 1], pred_all)
    return out
