# Opening the outputs in QGIS or ArcGIS

All spatial outputs are standard GeoTIFFs in EPSG:4326, so they open directly in QGIS, ArcGIS Pro, or any GDAL-based tool. No project file is shipped; building one takes a few minutes.

| File | Contents |
|---|---|
| `results/rasters/suitability_index.tif` | Continuous suitability, 0-1 |
| `results/rasters/suitability_classes.tif` | Classes: 1 very low, 2 low, 3 moderate, 4 high; 0 = no data |
| `data/processed/environmental_stack.tif` | All predictor layers as named bands |

## QGIS styling

1. **Layer > Add Layer > Add Raster Layer** and add `suitability_classes.tif`.
2. Open **Properties > Symbology**, set **Render type** to *Paletted/Unique values*, click **Classify**.
3. Set colours and labels:

| Value | Label | Colour |
|---|---|---|
| 4 | High | `#1a7f37` |
| 3 | Moderate | `#8bc34a` |
| 2 | Low | `#f2d600` |
| 1 | Very low | `#e8452c` |
| 0 | No data | transparent |

4. Add `data/processed/occurrences_clean.csv` via **Layer > Add Layer > Add Delimited Text Layer** (X field `lon`, Y field `lat`, CRS EPSG:4326) to overlay the presence records.
5. Save with **Project > Save As** to keep your own `.qgz`.

For the continuous surface, use *Singleband pseudocolour* with the *RdYlGn* ramp, minimum 0 and maximum 1.

## Area statistics

`results/class_area.csv` already holds the area per class, computed on the sphere so cell sizes are correct across latitude. Counting pixels in QGIS on an EPSG:4326 raster would not give square kilometres without first accounting for cell area.
