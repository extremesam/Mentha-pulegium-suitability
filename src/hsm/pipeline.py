"""End-to-end orchestration of the seven workflow stages."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from . import __version__
from .data_collection import fetch_gbif_occurrences, load_occurrence_csv
from .evaluation import cv_threshold, evaluate
from .features import build_dataset, prune_correlated
from .grid import Grid
from .integration import LayerStack
from .io import read_raster, write_raster
from .mapping import (
    CLASS_NAMES,
    area_by_class,
    classify,
    plot_classified_map,
    plot_cleaning_funnel,
    plot_continuous_map,
    plot_importance,
    plot_input_layers,
    plot_response_curves,
    plot_roc,
    plot_mess_panel,
    plot_transfer_map,
    plot_truth_vs_prediction,
)
from .modeling import (
    ensemble_predict,
    fit_models,
    make_models,
    predict_df,
    predict_surface,
    response_curves,
    spatial_block_cv,
    split_train_test,
    variable_importance,
)
from .projection import (
    clamp_stack,
    compute_mess,
    mask_from_geojson,
    project_models,
    transfer_area_table,
    variable_range_report,
)
from .processing import align_to_grid, clean_occurrences, derive_slope, fill_gaps, ndvi_composite
from .synthetic import generate_demo_data, generate_target_region

log = logging.getLogger("hsm")

CLASS_NAMES_ORDERED = CLASS_NAMES


def load_config(path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _j(obj):
    """JSON encoder for numpy scalars."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(type(obj))


def run(cfg: dict, mode: str = "demo", root: str | Path = ".") -> dict:
    """Run all seven stages and return a metrics dictionary (also written to results/)."""
    t0 = time.time()
    root = Path(root)
    paths = {k: root / v for k, v in cfg["paths"].items()}
    results = paths["results_dir"]
    fig_dir, ras_dir = results / "figures", results / "rasters"
    for d in (results, fig_dir, ras_dir, paths["processed_dir"]):
        d.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["project"]["seed"])
    species = cfg["project"]["species"]
    grid = Grid.from_config(cfg["study_area"])
    log.info("hsm %s | mode=%s | %s | grid %dx%d @ %.4f deg", __version__, mode, species, grid.height, grid.width, grid.resolution)

    # ------------------------------------------------------------------ 1. collection
    log.info("[1/7] data collection")
    truth = None
    truth_target = None
    if mode == "demo":
        need_scenes = any(spec.get("from_scenes") for spec in cfg["layers"].values())
        truth = generate_demo_data(grid, paths["demo_dir"], seed=seed, n_scenes=5 if need_scenes else 0)
        if cfg.get("projection", {}).get("enabled"):
            pcfg = cfg["projection"]
            truth_target = generate_target_region(
                Grid.from_config(pcfg["grid"]), paths["demo_dir"], seed=seed,
                warm=float(pcfg.get("demo_warm_shift", 5.0)), wet=float(pcfg.get("demo_wet_shift", 350.0)),
            )
        layers_dir = paths["demo_dir"] / "source_layers"
        scenes_dir = paths["demo_dir"] / "s2_scenes"
        occ_raw = load_occurrence_csv(paths["demo_dir"] / "occurrences_raw.csv")
    else:
        layers_dir = paths["raw_dir"] / "layers"
        scenes_dir = paths["raw_dir"] / "s2_scenes"
        oc = cfg["occurrences"]
        if oc.get("source", "gbif") == "csv":
            occ_raw = load_occurrence_csv(root / oc["csv"])
        else:
            occ_raw = fetch_gbif_occurrences(species, grid.bbox, limit=int(oc.get("max_gbif_records", 5000)))
            occ_raw.to_csv(paths["raw_dir"] / "occurrences_gbif.csv", index=False)

    # ------------------------------------------------------------------ 2. processing
    log.info("[2/7] data processing")
    oc = cfg["occurrences"]
    occ, steps = clean_occurrences(
        occ_raw,
        grid,
        max_uncertainty_m=oc.get("max_uncertainty_m"),
        year_min=oc.get("year_min"),
        thin_to_grid=oc.get("thin_to_grid", True),
    )
    for step, n in steps:
        log.info("   %-34s %6d", step, n)
    occ.to_csv(paths["processed_dir"] / "occurrences_clean.csv", index=False)
    plot_cleaning_funnel(steps, fig_dir / "01_occurrence_cleaning.png")

    layer_arrays: dict[str, np.ndarray] = {}
    ndvi_obs = None
    for name, spec in cfg["layers"].items():
        if "file" in spec:
            layer_arrays[name] = align_to_grid(layers_dir / spec["file"], grid, spec.get("resampling", "bilinear"))
        elif spec.get("from_scenes"):
            files = sorted(Path(scenes_dir).glob("*.tif"))
            if not files:
                raise FileNotFoundError(f"no Sentinel-2 scenes found in {scenes_dir}")
            scenes = []
            for f in files:
                red = align_to_grid(f, grid, "bilinear", band=1)
                nir = align_to_grid(f, grid, "bilinear", band=2)
                scl = align_to_grid(f, grid, "nearest", band=3)
                scenes.append((red, nir, np.nan_to_num(scl, nan=0).astype(int)))
            composite, ndvi_obs = ndvi_composite(scenes)
            composite, n_filled = fill_gaps(composite)
            layer_arrays[name] = composite
            log.info(
                "   NDVI composite from %d scenes (median clear obs/pixel: %d; %d cloud-gap pixels filled from nearest valid)",
                len(files), int(np.median(ndvi_obs)), n_filled,
            )
    for name, spec in cfg["layers"].items():
        if "derived_from" in spec:
            layer_arrays[name] = derive_slope(layer_arrays[spec["derived_from"]], grid)

    # ------------------------------------------------------------------ 3. integration
    log.info("[3/7] data integration")
    stack = LayerStack.from_layers({n: layer_arrays[n] for n in cfg["layers"]}, grid)
    stack.save(paths["processed_dir"] / "environmental_stack.tif")
    stack.summary().round(3).to_csv(results / "layer_summary.csv", index=False)
    plot_input_layers(stack, fig_dir / "02_environmental_layers.png")

    # ------------------------------------------------------------------ 4. features
    log.info("[4/7] feature extraction")
    data = build_dataset(
        stack,
        occ,
        n_background=int(cfg["background"]["n_points"]),
        seed=seed,
        exclude_presence_cells=cfg["background"].get("exclude_presence_cells", False),
        buffer_km=cfg["background"].get("buffer_km"),
    )
    features, dropped = prune_correlated(data, stack.names, cfg["features"]["correlation_threshold"])
    for d, partner, r in dropped:
        log.info("   dropped %s (|r|=%.2f with %s)", d, r, partner)
    data.to_csv(paths["processed_dir"] / "modeling_dataset.csv", index=False)
    n_pres, n_bg = int((data.label == 1).sum()), int((data.label == 0).sum())
    log.info("   %d presences, %d background points, %d predictors", n_pres, n_bg, len(features))

    # ------------------------------------------------------------------ 5. modelling
    log.info("[5/7] modelling")
    mc = cfg["modeling"]
    train, test = split_train_test(data, mc["test_size"], seed)
    models = make_models(mc, seed)
    fitted = fit_models(models, train, features)

    # ------------------------------------------------------------------ 6. evaluation
    log.info("[6/7] model evaluation")
    surfaces = {
        n: predict_surface(lambda X, m=m: m.predict_proba(X)[:, 1], stack, features)
        for n, m in fitted.items()
    }
    metrics: dict[str, dict] = {}
    roc_data: dict[str, tuple] = {}
    X_tr, y_tr = train[features].to_numpy(), train["label"].to_numpy()
    for name, m in fitted.items():
        p_test = predict_df(m, test, features)
        thr = cv_threshold(models[name], X_tr, y_tr)
        metrics[name] = evaluate(test["label"].to_numpy(), p_test, thr, pred_all=surfaces[name][stack.valid_mask])
        roc_data[name] = (test["label"].to_numpy(), p_test)
        if mc.get("spatial_cv", {}).get("enabled", True):
            sc = mc["spatial_cv"]
            aucs = spatial_block_cv(models[name], data, features, sc.get("block_deg", 2.0), sc.get("n_splits", 5))
            metrics[name]["spatial_cv_auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
            metrics[name]["spatial_cv_auc_sd"] = float(np.std(aucs)) if aucs else float("nan")
        log.info("   %-14s AUC %.3f | TSS %.3f | kappa %.3f", name, metrics[name]["auc"], metrics[name]["tss"], metrics[name]["kappa"])

    members = [n for n in fitted if metrics[n]["auc"] >= mc.get("ensemble_min_auc", 0.7)] or list(fitted)
    p_ens_test = ensemble_predict(fitted, test, features, members)
    ens_surface = np.mean([surfaces[m] for m in members], axis=0).astype("float32")
    ens_oof_thr = float(np.mean([metrics[m]["threshold"] for m in members]))
    metrics["ensemble"] = evaluate(test["label"].to_numpy(), p_ens_test, ens_oof_thr, pred_all=ens_surface[stack.valid_mask])
    metrics["ensemble"]["members"] = members
    roc_data["ensemble"] = (test["label"].to_numpy(), p_ens_test)
    log.info("   %-14s AUC %.3f | TSS %.3f | kappa %.3f  (members: %s)", "ensemble", metrics["ensemble"]["auc"], metrics["ensemble"]["tss"], metrics["ensemble"]["kappa"], ", ".join(members))

    imp_model = "random_forest" if "random_forest" in fitted else members[0]
    imp = variable_importance(fitted[imp_model], test, features, seed)
    imp.round(4).to_csv(results / "variable_importance.csv", index=False)

    ens_predictor = lambda X: np.mean([fitted[m].predict_proba(X)[:, 1] for m in members], axis=0)  # noqa: E731
    curves = response_curves(ens_predictor, data, features, seed=seed)

    comp = pd.DataFrame(metrics).T.drop(columns=["members"], errors="ignore")
    comp.round(4).to_csv(results / "model_comparison.csv")

    plot_roc(roc_data, {k: v["auc"] for k, v in metrics.items()}, fig_dir / "03_roc_curves.png")
    plot_importance(imp, fig_dir / "04_variable_importance.png", imp_model.replace("_", " "))
    plot_response_curves(curves, fig_dir / "05_response_curves.png")

    # ------------------------------------------------------------------ 7. mapping
    log.info("[7/7] suitability mapping")
    thr = cfg["mapping"]["class_thresholds"]
    classes = classify(ens_surface, thr)
    area = area_by_class(classes, grid)
    area.to_csv(results / "class_area.csv", index=False)
    write_raster(ras_dir / "suitability_index.tif", ens_surface, grid)
    write_raster(ras_dir / "suitability_classes.tif", classes, grid, nodata=0, dtype="uint8")
    title = f"Predicted habitat suitability - {species}"
    if mode == "demo":
        title += "\n(DEMO: virtual species on synthetic layers)"
    plot_continuous_map(ens_surface, grid, fig_dir / "06_suitability_continuous.png", title)
    plot_classified_map(classes, grid, fig_dir / "07_suitability_classes.png", title, occ)

    if truth is not None:
        both = ~np.isnan(ens_surface)
        rho = float(spearmanr(truth[both], ens_surface[both]).correlation)
        metrics["demo_validation"] = {"spearman_rho_vs_true_suitability": rho}
        plot_truth_vs_prediction(truth, ens_surface, grid, rho, fig_dir / "08_demo_truth_vs_prediction.png")
        log.info("   demo check: Spearman rho(true, predicted) = %.3f", rho)

    projection_summary = None
    if cfg.get("projection", {}).get("enabled"):
        log.info("[8/8] transfer projection")
        projection_summary = _run_projection(
            cfg, mode, root, paths, fitted, members, features, data, ras_dir, fig_dir, truth_target, species, thr
        )

    summary = {
        "species": species,
        "mode": mode,
        "version": __version__,
        "seed": seed,
        "grid": {"rows": grid.height, "cols": grid.width, "resolution_deg": grid.resolution, "bbox": grid.bbox},
        "records": {"raw": steps[0][1], "clean": len(occ), "presence_in_model": n_pres, "background": n_bg},
        "predictors_used": features,
        "predictors_dropped_collinear": [{"dropped": d, "kept": p, "abs_r": r} for d, p, r in dropped],
        "metrics": metrics,
        "class_area": area.to_dict(orient="records"),
        "projection": projection_summary,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(results / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=_j)
    log.info("done in %.1fs - outputs in %s", time.time() - t0, results)
    return summary


def _run_projection(cfg, mode, root, paths, fitted, members, features, data, ras_dir, fig_dir, truth_target, species, thr):
    """Stage 8: apply the trained ensemble to a new region and report how far to trust it."""
    pc = cfg["projection"]
    results = paths["results_dir"]
    tgrid = Grid.from_config(pc["grid"])
    name = pc.get("name", "target region")
    layers_dir = (paths["demo_dir"] / "target_layers") if mode == "demo" else Path(root / pc.get("layers_dir", cfg["paths"]["raw_dir"] + "/layers"))

    bad = [f for f in features if cfg["layers"][f].get("from_scenes")]
    if bad:
        raise ValueError(
            f"predictor(s) {bad} come from satellite scenes and cannot be transferred to another region. "
            "Remove them from `layers:` in the config for a projection run."
        )

    arrays: dict[str, np.ndarray] = {}
    for lname, spec in cfg["layers"].items():
        if "file" in spec:
            arrays[lname] = align_to_grid(layers_dir / spec["file"], tgrid, spec.get("resampling", "bilinear"))
    for lname, spec in cfg["layers"].items():
        if "derived_from" in spec:
            arrays[lname] = derive_slope(arrays[spec["derived_from"]], tgrid)
    tstack = LayerStack.from_layers({n: arrays[n] for n in features}, tgrid)

    if pc.get("mask_geojson"):
        inside = mask_from_geojson(root / pc["mask_geojson"], tgrid)
        tstack.data[:, ~inside] = np.nan
        log.info("   masked to %s boundary: %d of %d cells kept", name, int(inside.sum()), inside.size)

    reference = data[features]
    mess, mod = compute_mess(reference, tstack)
    in_domain = np.nan_to_num(mess, nan=-1.0) >= 0
    valid = ~np.isnan(mess)
    pct_in = float(100 * in_domain[valid].sum() / valid.sum())
    report = variable_range_report(reference, tstack)
    report.round(3).to_csv(results / "transfer_variable_ranges.csv", index=False)

    work = clamp_stack(tstack, reference) if pc.get("clamp", True) else tstack
    mean, sd, _ = project_models(fitted, members, work, features)
    classes = classify(mean, thr)
    area = transfer_area_table(classes, in_domain, tgrid, CLASS_NAMES_ORDERED)
    area.to_csv(results / "transfer_class_area.csv", index=False)

    write_raster(ras_dir / "transfer_suitability_index.tif", mean, tgrid)
    write_raster(ras_dir / "transfer_suitability_classes.tif", classes, tgrid, nodata=0, dtype="uint8")
    write_raster(ras_dir / "transfer_mess.tif", mess, tgrid)
    write_raster(ras_dir / "transfer_model_sd.tif", sd, tgrid)
    write_raster(ras_dir / "transfer_suitability_in_domain.tif", np.where(in_domain, mean, np.nan).astype("float32"), tgrid)

    title = f"Transfer suitability - {species} in {name}"
    if mode == "demo":
        title += "\n(DEMO: virtual species, synthetic target region)"
    plot_transfer_map(classes, in_domain, tgrid, fig_dir / "09_transfer_suitability.png", title)
    plot_mess_panel(mess, mod, sd, features, tgrid, fig_dir / "10_transfer_extrapolation_diagnostics.png")

    log.info("   %s: %.1f%% of cells inside the training climate (MESS >= 0)", name, pct_in)
    for _, r in report.sort_values("pct_target_outside_train", ascending=False).head(3).iterrows():
        log.info("   %-10s %.1f%% of cells outside training range", r["variable"], r["pct_target_outside_train"])

    out = {
        "name": name,
        "grid": {"rows": tgrid.height, "cols": tgrid.width, "resolution_deg": tgrid.resolution, "bbox": tgrid.bbox},
        "pct_cells_in_training_domain": round(pct_in, 2),
        "clamped": bool(pc.get("clamp", True)),
        "members": members,
        "mean_suitability_all": float(np.nanmean(mean)),
        "mean_suitability_in_domain": float(np.nanmean(np.where(in_domain, mean, np.nan))) if in_domain.any() else None,
        "mean_model_sd": float(np.nanmean(sd)),
        "class_area": area.to_dict(orient="records"),
        "variable_ranges": report.round(3).to_dict(orient="records"),
    }
    if truth_target is not None:
        both = valid & ~np.isnan(mean)
        rho_all = float(spearmanr(truth_target[both], mean[both]).correlation)
        res = {"spearman_all": rho_all, "mean_abs_error_all": float(np.mean(np.abs(truth_target[both] - mean[both])))}
        for tag, sel in (("in_domain", both & in_domain), ("extrapolated", both & ~in_domain)):
            if sel.sum() > 50 and np.std(truth_target[sel]) > 0:
                res[f"spearman_{tag}"] = float(spearmanr(truth_target[sel], mean[sel]).correlation)
                res[f"mean_abs_error_{tag}"] = float(np.mean(np.abs(truth_target[sel] - mean[sel])))
                res[f"n_cells_{tag}"] = int(sel.sum())
        out["demo_validation"] = res
        plot_truth_vs_prediction(truth_target, mean, tgrid, rho_all, fig_dir / "11_transfer_truth_vs_prediction.png", label="Transfer validation")
        log.info("   demo transfer check: Spearman rho(true, predicted) = %.3f", rho_all)
    return out
