#!/usr/bin/env python3
"""Fetch and prepare the real input layers for the Nigeria transfer run.

    python scripts/prepare_real_data.py worldclim [--res 5m] [--out data/raw/layers]
    python scripts/prepare_real_data.py boundary  [--iso NGA] [--out data/raw/nigeria_adm0.geojson]

``worldclim`` downloads the WorldClim 2.1 bioclimatic + elevation archives, extracts the
variables the config uses, and renames them to ``bio1.tif`` ... ``elevation_dem.tif``.
Pass ``--bio-zip`` / ``--elev-zip`` to use archives you downloaded yourself (also how the
unit test exercises this script without network access).

STATUS: the extraction and renaming logic is tested offline against a synthetic archive.
The download URLs follow the pattern published on worldclim.org and the geoBoundaries API,
but have NOT been exercised from the environment this was written in. If a download fails,
fetch the file in a browser and pass it with --bio-zip / --elev-zip (or --geojson).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

WORLDCLIM_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"
DEFAULT_BIOS = (1, 5, 6, 12, 15, 17)


def download(url: str, dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return dest


def extract_worldclim(bio_zip: Path, elev_zip: Path, out_dir: Path, bios=DEFAULT_BIOS) -> list[Path]:
    """Pull the wanted bio layers and elevation out of the archives under the names the config expects."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def pick(zf: zipfile.ZipFile, pattern: str):
        rx = re.compile(pattern, re.IGNORECASE)
        return [n for n in zf.namelist() if rx.search(n)]

    with zipfile.ZipFile(bio_zip) as zf:
        for b in bios:
            hits = pick(zf, rf"bio[_-]?0*{b}\.tif$")
            if not hits:
                raise FileNotFoundError(f"bio{b} not found in {bio_zip.name}; archive contains: {zf.namelist()[:5]} ...")
            dest = out_dir / f"bio{b}.tif"
            with zf.open(hits[0]) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written.append(dest)

    with zipfile.ZipFile(elev_zip) as zf:
        hits = pick(zf, r"elev[^/]*\.tif$")
        if not hits:
            raise FileNotFoundError(f"no elevation .tif in {elev_zip.name}")
        dest = out_dir / "elevation_dem.tif"
        with zf.open(hits[0]) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
        written.append(dest)
    return written


def cmd_worldclim(a) -> int:
    out = Path(a.out)
    cache = Path(a.cache)
    bio_zip = Path(a.bio_zip) if a.bio_zip else download(f"{WORLDCLIM_BASE}/wc2.1_{a.res}_bio.zip", cache / f"wc2.1_{a.res}_bio.zip")
    elev_zip = Path(a.elev_zip) if a.elev_zip else download(f"{WORLDCLIM_BASE}/wc2.1_{a.res}_elev.zip", cache / f"wc2.1_{a.res}_elev.zip")
    for p in extract_worldclim(bio_zip, elev_zip, out, tuple(a.bios)):
        print("wrote", p)
    print("\nCite: Fick, S.E. and Hijmans, R.J. (2017) WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. Int. J. Climatol. 37: 4302-4315.")
    return 0


def cmd_boundary(a) -> int:
    import requests

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.geojson:
        shutil.copyfile(a.geojson, out)
        print("copied", out)
        return 0
    api = f"https://www.geoboundaries.org/api/current/gbOpen/{a.iso}/ADM0/"
    meta = requests.get(api, timeout=60)
    meta.raise_for_status()
    info = meta.json()
    if isinstance(info, list):
        info = info[0]
    url = info.get("gjDownloadURL") or info.get("simplifiedGeometryGeoJSON")
    if not url:
        raise RuntimeError(f"geoBoundaries response had no GeoJSON URL. Keys: {sorted(info)}")
    download(url, out)
    print("wrote", out)
    print("\nCite geoBoundaries (Runfola et al. 2020) and check the licence shown at geoboundaries.org.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("worldclim", help="download + extract WorldClim 2.1 bio and elevation")
    w.add_argument("--res", default="5m", choices=["10m", "5m", "2.5m"])
    w.add_argument("--out", default="data/raw/layers")
    w.add_argument("--cache", default="data/raw/downloads")
    w.add_argument("--bios", type=int, nargs="+", default=list(DEFAULT_BIOS))
    w.add_argument("--bio-zip")
    w.add_argument("--elev-zip")
    w.set_defaults(fn=cmd_worldclim)

    b = sub.add_parser("boundary", help="fetch a country outline (GeoJSON) for masking")
    b.add_argument("--iso", default="NGA")
    b.add_argument("--out", default="data/raw/nigeria_adm0.geojson")
    b.add_argument("--geojson", help="use a GeoJSON you already downloaded instead of the API")
    b.set_defaults(fn=cmd_boundary)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
