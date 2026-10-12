"""Stage 7 - suitability mapping: classification, area statistics and figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .grid import Grid  # noqa: E402

CLASS_NAMES = {1: "Very low", 2: "Low", 3: "Moderate", 4: "High"}
CLASS_COLOURS = {1: "#e8452c", 2: "#f2d600", 3: "#8bc34a", 4: "#1a7f37"}
NODATA_CLASS = 0


def classify(surface: np.ndarray, thresholds=(0.25, 0.5, 0.75)) -> np.ndarray:
    """Bin a 0-1 suitability surface into classes 1-4 (0 = no data)."""
    out = np.full(surface.shape, NODATA_CLASS, dtype="uint8")
    ok = ~np.isnan(surface)
    out[ok] = 1 + np.digitize(surface[ok], thresholds)
    return out


def area_by_class(classes: np.ndarray, grid: Grid) -> pd.DataFrame:
    area = grid.cell_area_km2()
    total = area[classes != NODATA_CLASS].sum()
    rows = []
    for k in (4, 3, 2, 1):
        km2 = float(area[classes == k].sum())
        rows.append({"class": CLASS_NAMES[k], "area_km2": round(km2, 1), "percent": round(100 * km2 / total, 2)})
    return pd.DataFrame(rows)


def _extent(grid: Grid):
    return (grid.west, grid.east, grid.south, grid.north)


def _finish(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_classified_map(classes, grid: Grid, path, title, points: pd.DataFrame | None = None):
    cmap = ListedColormap(["white"] + [CLASS_COLOURS[k] for k in (1, 2, 3, 4)])
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.imshow(classes, cmap=cmap, vmin=0, vmax=4, extent=_extent(grid), interpolation="nearest")
    if points is not None and len(points):
        ax.scatter(points["lon"], points["lat"], s=7, c="black", alpha=0.55, linewidths=0, label="Presence records")
    handles = [Patch(color=CLASS_COLOURS[k], label=CLASS_NAMES[k]) for k in (4, 3, 2, 1)]
    if points is not None and len(points):
        handles.append(plt.Line2D([], [], marker="o", ls="", color="black", ms=4, label="Presence records"))
    ax.legend(handles=handles, title="Suitability", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return _finish(fig, path)


def plot_continuous_map(surface, grid: Grid, path, title):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    im = ax.imshow(surface, cmap="RdYlGn", vmin=0, vmax=1, extent=_extent(grid), interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Relative suitability index (0-1)")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return _finish(fig, path)


def plot_input_layers(stack, path):
    n = len(stack.names)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, name, band in zip(axes, stack.names, stack.data):
        im = ax.imshow(band, cmap="viridis", extent=_extent(stack.grid))
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Environmental layer stack (common grid)", weight="bold")
    return _finish(fig, path)


def plot_roc(roc_data: dict[str, tuple[np.ndarray, np.ndarray]], aucs: dict[str, float], path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for name, (y, p) in roc_data.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name.replace('_', ' ')} (AUC {aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - 30 % hold-out test set", weight="bold")
    ax.legend(loc="lower right", fontsize=8)
    return _finish(fig, path)


def plot_importance(imp: pd.DataFrame, path, model_name="random forest"):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    d = imp.iloc[::-1]
    ax.barh(d["variable"], d["importance"], xerr=d["std"], color="#2e7d32")
    ax.set_xlabel("Drop in test AUC when shuffled")
    ax.set_title(f"Permutation importance ({model_name})", weight="bold")
    return _finish(fig, path)


def plot_response_curves(curves: dict[str, pd.DataFrame], path):
    n = len(curves)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.7 * rows), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, c) in zip(axes, curves.items()):
        ax.plot(c["value"], c["suitability"], color="#1a7f37", lw=2)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Ensemble response curves (partial dependence)", weight="bold")
    return _finish(fig, path)


def plot_cleaning_funnel(steps: list[tuple[str, int]], path):
    labels = [s for s, _ in steps][::-1]
    counts = [c for _, c in steps][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.barh(labels, counts, color="#1565c0")
    for b, c in zip(bars, counts):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {c}", va="center", fontsize=9)
    ax.set_xlabel("Records remaining")
    ax.set_title("Occurrence data cleaning", weight="bold")
    ax.set_xlim(0, max(counts) * 1.12)
    return _finish(fig, path)


def plot_truth_vs_prediction(truth, surface, grid: Grid, rho: float, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, data, title in zip(axes, (truth, surface), ("Virtual species: true suitability", "Model: predicted suitability")):
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, extent=_extent(grid), interpolation="nearest")
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Suitability (0-1)")
    fig.suptitle(f"Demo validation: Spearman rho = {rho:.3f}", y=1.0)
    return _finish(fig, path)
