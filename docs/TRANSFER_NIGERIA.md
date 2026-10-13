# Projecting suitability onto Nigeria

GBIF shows no *Mentha pulegium* records in Nigeria. So the question here is not "where does it grow in Nigeria?" but:

> **Where in Nigeria are the climate and terrain similar to the places this species is known to live, and how far should that answer be trusted?**

The pipeline's stage 8 answers it. It trains on the species' known range elsewhere, predicts onto a Nigeria grid, and reports how much of that prediction is extrapolation.

## What the output does and does not mean

| It means | It does not mean |
|---|---|
| Conditions in this cell resemble those at known records | The species grows, or would establish, here |
| Where the model is extrapolating (grey on the map) | That extrapolated cells are unsuitable. They are *unknown* |
| Which models disagree | A calibrated probability |

Dispersal, land use, competition, soil, microhabitat (the species favours seasonally wet ground) and human introduction are all outside the model.

## Run it

Needs Python 3.10+. If you work in the browser, **Google Colab** or **GitHub Codespaces** both work.

```bash
git clone https://github.com/YOUR-USERNAME/mentha-habitat-suitability.git
cd mentha-habitat-suitability
pip install -e .

python scripts/prepare_real_data.py worldclim          # WorldClim 2.1 bio + elevation (5 arc-min)
python scripts/prepare_real_data.py boundary --iso NGA # Nigeria outline for masking and area stats
hsm run --config config/nigeria_transfer.yaml --mode real
```

In a Colab notebook, put `!` before each command and `%cd mentha-habitat-suitability` after the clone. To download the results:

```python
!zip -r results_nigeria.zip results_nigeria
from google.colab import files; files.download("results_nigeria.zip")
```

**Status of this path.** The projection code, MESS, clamping, masking and the data-prep script's extraction logic are unit-tested, and the full transfer runs end to end on synthetic data. **The live downloads (GBIF, WorldClim, geoBoundaries) have not been run from where this was written.** The WorldClim URLs match the ones published on worldclim.org. The geoBoundaries call and the file names inside the WorldClim archives are assumptions. If a download fails, fetch the file in a browser and pass it to the script with `--bio-zip`, `--elev-zip` or `--geojson`. I have not timed the real run (about 740,000 grid cells for the training area), so expect minutes to tens of minutes.

## Design choices

- **Train worldwide, not just on the native range.** Introduced populations (e.g. Australia, the Americas) show the species in a wider range of climates. That gives the model more to go on when predicting somewhere warmer.
- **Climate and elevation only.** Soil maps carry large regional uncertainty, and NDVI describes a particular scene rather than climate suitability, so the config leaves both out. The pipeline refuses scene-derived predictors in a projection run.
- **Same resolution for training and projection** (0.25° ≈ 28 km). Climate averaged over big cells and climate sampled on small cells have different variance, so mixing scales biases the result. A 0.25° map of Nigeria is only about 50 × 40 cells. Finer cells are possible but should be treated as a preview.
- **Clamping.** Predictors in the target are clipped to the training range before predicting, so the regularised quadratic model cannot run away outside the data.
- **MESS** (Elith et al. 2010) is computed against the presence + background rows that trained the models. Negative = at least one predictor is outside the training range. The *most dissimilar variable* map says which one.
- **Model disagreement** is the standard deviation across the three algorithms. It is highest where suitability itself is moderate or high, so read it together with the suitability map, not in place of it.

## Output files (`results_nigeria/`)

| File | Use it for |
|---|---|
| `figures/09_transfer_suitability.png` | The headline map, with extrapolated cells greyed |
| `figures/10_transfer_extrapolation_diagnostics.png` | MESS, which variable is out of range, model disagreement |
| `rasters/transfer_suitability_index.tif` | Continuous 0-1 index for GIS |
| `rasters/transfer_suitability_in_domain.tif` | Same, with extrapolated cells blank |
| `rasters/transfer_mess.tif`, `transfer_model_sd.tif` | Diagnostics as layers |
| `transfer_class_area.csv` | Area per class, split into inside / outside the training climate |
| `transfer_variable_ranges.csv` | Per predictor: training range vs Nigeria range, share of Nigeria outside |
| `metrics.json` | Everything above plus the training-model metrics |

Report **both** columns of `transfer_class_area.csv`. The "in domain" area is the defensible number.

## What I expect (a hypothesis, not a result)

As far as I know, *M. pulegium* is a temperate and Mediterranean species of seasonally wet ground, while most of Nigeria is hot, lowland tropics. I would expect much of the country to be flagged as outside the training climate or to score low. The cooler highlands (the Jos Plateau, Mambilla Plateau and Obudu area) are where I would look first for higher scores. If the real run shows something else, believe the run over this paragraph.

## Sensitivity runs to do before trusting anything

1. **Background:** run with `background.buffer_km: 1500` and with `null`. Background across the whole world teaches the model that the tropics are unsuitable. A buffer teaches it only the contrast near where the species lives. Large differences in Nigeria mean the answer depends on this choice.
2. **Training area:** swap `study_area` for a native-range box (west -12, south 28, east 50, north 55) and compare.
3. **Predictors:** drop `bio5`/`bio6`, or add others, and see which Nigerian cells keep their class.
4. **Seed and algorithm:** if the class map changes a lot between seeds or between models, say so.

## Validation

There are no verified absences in Nigeria, and the species is not recorded there, so a conventional accuracy test is impossible. What you can do:

- Report training-range diagnostics (spatial-block CV AUC, Boyce) as evidence the model works *where data exist*.
- Show the MESS map so a reader can see where it has no evidence.
- Compare with climate analogues: are there GBIF records of the species in the highlands of other tropical countries? Those are the closest test of transfer.
- Ask local botanists or herbarium staff whether anything similar is known from the highlands.

## Known limits

- GBIF records can include **cultivated** plants (mint is a garden and crop species). The pipeline does not filter on establishment means or basis of record.
- Records are sampling-biased towards Europe and North America. Thinning to one per cell helps but does not remove it.
- The 1970-2000 climate normals are historical; current and future climate differ.
- A 28 km cell hides the small wet depressions the species actually occupies.
