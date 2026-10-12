# data/raw

Put real downloads here for `--mode real` (this folder is git-ignored except for this file):

```
data/raw/
├── occurrences.csv          # optional, if occurrences.source: csv  (lon, lat [, year, coordinateUncertaintyInMeters])
├── layers/
│   ├── bio1.tif             # WorldClim mean annual temperature
│   ├── bio12.tif            # WorldClim annual precipitation
│   ├── bio15.tif            # WorldClim precipitation seasonality
│   ├── elevation_dem.tif    # SRTM DEM
│   ├── soil_ph.tif          # SoilGrids pH
│   └── clay.tif             # SoilGrids clay content
└── s2_scenes/
    └── *.tif                # Sentinel-2 L2A scenes: band 1 = B04 (red), band 2 = B08 (NIR), band 3 = SCL
```

File names, resampling methods and the predictor list are set in `config/config.yaml`.
See `docs/DATA_SOURCES.md` for where each layer comes from and how to cite it.
