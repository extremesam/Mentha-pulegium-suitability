# Methodology

## Problem framing

Habitat suitability modelling (HSM), also called species distribution modelling, relates known occurrence locations to environmental conditions and projects that relationship over a study area. The result shows where conditions resemble those at known occurrences. It is **not** a detection map and **not** proof of presence: dispersal history, biotic interactions and land use are not modelled.

## Stage 1: Data collection

| Input | Source | Used for |
|---|---|---|
| Species occurrences | GBIF | Presence records |
| Climate (bio1, bio12, bio15) | WorldClim 2.1 | Temperature and precipitation regime |
| Elevation | SRTM DEM | Elevation, slope |
| Soil (pH, clay) | SoilGrids | Edaphic niche |
| Imagery | Sentinel-2 L2A | Vegetation index (NDVI) |

See [DATA_SOURCES.md](DATA_SOURCES.md) for access and citation details.

## Stage 2: Processing

**Occurrence cleaning** removes, in order: records without coordinates; invalid or (0, 0) coordinates; records outside the study area; records whose stated coordinate uncertainty exceeds a threshold (default 5 km; unknown uncertainty is kept); records before a minimum year; exact duplicate coordinates; and finally keeps one record per grid cell. Each step's remaining count is logged and plotted.

**NDVI** is computed per scene as (NIR - Red) / (NIR + Red). Pixels classed by the Sentinel-2 Scene Classification Layer as no-data, saturated, cloud shadow, cloud (medium or high probability), thin cirrus or snow are masked. The per-pixel **median** of the clear observations forms the composite, which is robust to residual contamination. Pixels cloudy in every scene are filled from the nearest valid pixel and the count is logged; gather more scenes if that count is large.

**Resampling.** Every layer is clipped to the study extent and resampled onto one grid. `bilinear` is used for continuous layers whose source is coarser than or similar to the study grid; `average` when aggregating a finer raster (e.g. a DEM) down to it. Categorical rasters should use `nearest`.

**Slope** is derived from the aligned DEM using central differences with latitude-corrected cell width.

## Stage 3: Integration

All layers share one `Grid` (north-up, square cells, EPSG:4326) and are stacked into a `(layers, rows, cols)` cube with a combined validity mask. The stack is saved as a multi-band GeoTIFF with band descriptions.

## Stage 4: Feature extraction

Predictor values are sampled at presence cells. **Background points** (default 5,000) are drawn uniformly from valid cells. Predictors are then screened for collinearity: walking the predictor list in priority order, a variable is dropped if its absolute Pearson correlation with an already-kept variable exceeds 0.85. In the demo run, elevation was dropped for correlating at 0.88 with mean annual temperature.

## Stage 5: Models

| Model | Notes |
|---|---|
| Random forest | 500 trees, `min_samples_leaf=5`, balanced subsample class weights |
| MaxEnt-style | Standardised linear + quadratic terms, L1-penalised logistic regression, penalty strength chosen by 5-fold CV on AUC. Maxent and regularised Poisson regression are equivalent (Renner & Warton 2013), so this is a close analogue, not the Maxent Java program |
| Boosted trees | Histogram gradient boosting, balanced class weights |
| Ensemble | Unweighted mean of members whose test AUC is at least 0.70 |

Predictions are **relative suitability indices on 0-1**. With pseudo-absences, absolute probability of occurrence cannot be recovered; the index is meaningful for ranking, and class boundaries are conventions.

## Stage 6: Evaluation

- **Split:** stratified 70 % training / 30 % testing.
- **AUC:** threshold-free discrimination of presences from background.
- **TSS:** sensitivity + specificity - 1; unlike kappa, not affected by prevalence.
- **Kappa and accuracy:** reported for completeness; both depend heavily on the presence:background ratio.
- **Boyce index:** continuous Boyce (Hirzel et al. 2006) using test presences against the whole-area prediction; needs no absences. +1 is perfect, 0 random.
- **Threshold:** the TSS-maximising threshold is found on out-of-fold *training* predictions and then applied to the test set, so the test set is not used for tuning.
- **Spatial-block CV:** the study area is divided into 2-degree blocks and whole blocks are held out (GroupKFold), a stricter test of transfer than random splitting when records are spatially autocorrelated.
- **Variable importance:** permutation importance (drop in test AUC) for the random forest.
- **Response curves:** partial dependence of the ensemble over a 500-row background sample.

## Stage 7: Mapping

The ensemble surface is classified at 0.25 / 0.50 / 0.75 into *very low*, *low*, *moderate* and *high*. Class areas use spherical cell areas (km²), so they are correct across latitude. Outputs: continuous GeoTIFF, class GeoTIFF (0 = no data), area table and PNG maps.

## Demo data design

`synthetic.py` builds smooth correlated random fields for elevation, temperature (with a lapse rate), precipitation, seasonality, soil pH and clay, and an NDVI field tied to moisture and temperature. A **virtual species** is defined by a product of Gaussian responses to these variables (the approach of the R package `virtualspecies`). Presences are sampled in proportion to true suitability squared, then deliberately contaminated with missing coordinates, (0, 0) points, out-of-area points, coarse-uncertainty records, old records, exact duplicates and near-duplicates so the cleaning stage has work to do. Source rasters are written at three different resolutions, with a 1-degree buffer, so clipping and resampling are exercised. Cloudy Sentinel-2-like scenes are generated so masking and compositing run on real-shaped inputs.

Because the true niche is known, the demo reports Spearman correlation between true and predicted suitability as an end-to-end validation. That check is only possible in demo mode.
