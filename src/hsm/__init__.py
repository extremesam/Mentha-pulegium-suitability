"""hsm: a reproducible GIS habitat suitability modelling pipeline.

The package mirrors the seven stages of the project workflow:

1. data collection      -> :mod:`hsm.data_collection`
2. data processing      -> :mod:`hsm.processing`
3. data integration     -> :mod:`hsm.integration`
4. feature extraction   -> :mod:`hsm.features`
5. modelling            -> :mod:`hsm.modeling`
6. model evaluation     -> :mod:`hsm.evaluation`
7. suitability mapping  -> :mod:`hsm.mapping`

:mod:`hsm.pipeline` wires the stages together and :mod:`hsm.cli` exposes them
on the command line.
"""

__version__ = "1.0.0"
