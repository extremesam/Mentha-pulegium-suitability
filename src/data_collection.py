"""Stage 1 - data collection.

Species occurrences come from GBIF (live API) or a user-supplied CSV. Climate,
terrain, soil and imagery layers are downloaded separately (see
``docs/DATA_SOURCES.md``) because they are large and licence-bound.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

GBIF_API = "https://api.gbif.org/v1"

OCCURRENCE_COLUMNS = [
    "gbifID",
    "species",
    "lon",
    "lat",
    "coordinateUncertaintyInMeters",
    "year",
    "basisOfRecord",
    "datasetKey",
    "countryCode",
]


def match_taxon_key(name: str, session: requests.Session | None = None) -> int:
    """Resolve a scientific name to a GBIF backbone taxon key."""
    s = session or requests
    r = s.get(f"{GBIF_API}/species/match", params={"name": name}, timeout=30)
    r.raise_for_status()
    data = r.json()
    key = data.get("usageKey")
    if key is None:
        raise ValueError(f"GBIF could not match taxon name {name!r}: {data.get('note', data)}")
    return int(key)


def fetch_gbif_occurrences(
    species: str,
    bbox: tuple[float, float, float, float],
    limit: int = 5000,
    page_size: int = 300,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download georeferenced PRESENT records inside ``bbox`` (west, south, east, north).

    Uses the paginated search endpoint (hard cap of 100,000 records). For
    anything published, request a GBIF *download* instead so you receive a
    citable DOI.
    """
    s = session or requests
    west, south, east, north = bbox
    key = match_taxon_key(species, session=session)
    rows: list[dict] = []
    offset = 0
    while len(rows) < limit:
        params = {
            "taxonKey": key,
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "occurrenceStatus": "PRESENT",
            "decimalLongitude": f"{west},{east}",
            "decimalLatitude": f"{south},{north}",
            "limit": min(page_size, limit - len(rows)),
            "offset": offset,
        }
        r = s.get(f"{GBIF_API}/occurrence/search", params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results", [])
        for rec in results:
            rows.append(
                {
                    "gbifID": rec.get("gbifID"),
                    "species": rec.get("species") or rec.get("scientificName"),
                    "lon": rec.get("decimalLongitude"),
                    "lat": rec.get("decimalLatitude"),
                    "coordinateUncertaintyInMeters": rec.get("coordinateUncertaintyInMeters"),
                    "year": rec.get("year"),
                    "basisOfRecord": rec.get("basisOfRecord"),
                    "datasetKey": rec.get("datasetKey"),
                    "countryCode": rec.get("countryCode"),
                }
            )
        offset += len(results)
        if payload.get("endOfRecords", True) or not results:
            break
    log.info("GBIF returned %d records for %s", len(rows), species)
    return pd.DataFrame(rows, columns=OCCURRENCE_COLUMNS)


def load_occurrence_csv(path) -> pd.DataFrame:
    """Load occurrences from CSV. Accepts GBIF-style or simple lon/lat column names."""
    df = pd.read_csv(path, sep=None, engine="python")
    rename = {
        "decimalLongitude": "lon",
        "decimalLatitude": "lat",
        "longitude": "lon",
        "latitude": "lat",
        "x": "lon",
        "y": "lat",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns and v not in df.columns})
    missing = {"lon", "lat"} - set(df.columns)
    if missing:
        raise ValueError(f"{Path(path).name} is missing required column(s): {sorted(missing)}")
    return df
