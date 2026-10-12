# Data sources

Download these for `--mode real` and place them as described in [`data/raw/README.md`](../data/raw/README.md). Check each provider's current licence and citation requirements before publishing.

| Layer | Provider | Where | Notes |
|---|---|---|---|
| Species occurrences | GBIF | https://www.gbif.org | The pipeline can query the API directly. For publication, request a GBIF *download* so you get a citable DOI. |
| Climate: bio1, bio12, bio15 | WorldClim 2.1 | https://www.worldclim.org/data/worldclim21.html | Cite Fick & Hijmans (2017). Choose a resolution matching `study_area.resolution`. |
| Elevation | SRTM | https://earthexplorer.usgs.gov (search for SRTM 1 Arc-Second Global) | Void-filled tiles are easier to work with; mosaic and clip to your area. |
| Soil: pH, clay | SoilGrids | https://soilgrids.org | Cite Poggio et al. (2021). Pick a depth interval and use it consistently. |
| Imagery | Sentinel-2 L2A | https://dataspace.copernicus.eu | Export band 1 = B04 (red), band 2 = B08 (NIR), band 3 = SCL (scene classification), same grid per scene. Several scenes across the season, ideally low cloud. |

## Preparing Sentinel-2 scenes

The pipeline expects one GeoTIFF per scene with three bands in this order: **B04, B08, SCL**. Reflectance may be scaled by 10,000 (the NDVI ratio is unaffected). The SCL band is resampled with nearest neighbour; red and NIR with bilinear.

## Adding or swapping predictors

Add an entry under `layers:` in `config/config.yaml`:

```yaml
  bio6: {file: bio6.tif, resampling: bilinear}
```

List variables in priority order: the collinearity filter keeps earlier variables over later ones.

## Choosing predictors for a real study

The demo predictor set is illustrative. For a real analysis, choose variables on ecological grounds for the species and region (for a plant associated with seasonally wet ground, for example, moisture and seasonality variables are natural candidates), and document why.
