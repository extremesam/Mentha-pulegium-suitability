"""Command-line interface: ``hsm run --config config/config.yaml --mode demo``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import load_config, run


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hsm", description="GIS habitat suitability modelling pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run all seven stages")
    r.add_argument("--config", default="config/config.yaml")
    r.add_argument("--mode", choices=["demo", "real"], default="demo",
                   help="demo = synthetic virtual-species data (no downloads); real = your data in data/raw/")
    r.add_argument("--root", default=".", help="project root that config paths are relative to")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    run(cfg, mode=args.mode, root=Path(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
