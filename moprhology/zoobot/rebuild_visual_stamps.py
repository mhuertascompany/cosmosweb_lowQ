"""Regenerate COSMOS-Web visual morphology stamps using RA/Dec stored in the visual catalog.

This script mirrors the cutting utilities in ``make_stamps.py`` but operates directly on
the visual morphology SQLite database. It parses the RA/Dec/tile string, loads the
appropriate mosaic for each requested filter, and saves JPEG stamps into per-filter
directories under the chosen output root (e.g., ``/n03data/.../stamps_ilbert/F150W``).

Example:

```
python -m moprhology.zoobot.rebuild_visual_stamps \
    --visual-db /n07data/ilbert/COSMOS-Web/photoz_MASTER_v3.1.0/MORPHO/visualmorpho_COSMOSWeb_v7.db \
    --table morphology \
    --coord-column INFO \
    --filters F115W F150W F277W F444W \
    --output-root /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \
    --samples-per-filter 999999
```
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from skimage.transform import resize
from moprhology.zoobot.make_stamps import (
    load_imgs,
    image_make_cutout,
    array2img,
    zero_pix_fraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate COSMOS-Web visual morphology stamps.")
    parser.add_argument("--visual-db", type=Path, required=True, help="Path to the visual morphology SQLite DB.")
    parser.add_argument("--table", default="morphology", help="Table name inside the SQLite DB.")
    parser.add_argument("--coord-column", default="INFO", help="Column containing 'id RA DEC tile' strings.")
    parser.add_argument("--filters", nargs="+", default=["F115W", "F150W", "F277W", "F444W"], help="List of filters to cut.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory to store per-filter stamp folders.")
    parser.add_argument("--arcsec-size", type=float, default=2.5, help="Cutout half-size in arcseconds.")
    parser.add_argument("--resize-px", type=int, default=424, help="Output stamp size (square) in pixels.")
    parser.add_argument("--max-objects", type=int, default=None, help="Optional limit on number of objects processed.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when limiting objects.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JPEGs.")
    return parser.parse_args()


def load_visual_catalog(db_path: Path, table: str, coord_column: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = f"SELECT id, {coord_column} FROM {table}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if coord_column not in df.columns:
        raise ValueError(f"Column '{coord_column}' not found in table '{table}'.")
    return df


def parse_coord(entry: str) -> Tuple[str, float, float, str]:
    if entry is None:
        raise ValueError("Empty coordinate entry")
    tokens = str(entry).strip().split()
    if len(tokens) < 4:
        raise ValueError(f"Cannot parse coordinate string '{entry}'")
    obj_id = tokens[0]
    ra = float(tokens[1])
    dec = float(tokens[2])
    tile = tokens[3]
    return obj_id, ra, dec, tile


def ensure_output_dirs(root: Path, filters: Iterable[str]) -> None:
    for filt in filters:
        (root / filt).mkdir(parents=True, exist_ok=True)


def save_stamp(image: np.ndarray, path: Path, resize_px: int) -> None:
    scaled = resize(image, output_shape=(resize_px, resize_px))
    array2img(scaled).save(path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    df = load_visual_catalog(args.visual_db, args.table, args.coord_column)
    if args.max_objects is not None and args.max_objects < len(df):
        df = df.sample(n=args.max_objects, random_state=args.seed)

    ensure_output_dirs(args.output_root, args.filters)

    success = 0
    missing = 0

    for row in df.itertuples(index=False):
        try:
            coord_id, ra, dec, tile = parse_coord(getattr(row, args.coord_column))
        except Exception as exc:
            logging.warning("Skipping row due to coord parse failure: %s", exc)
            continue

        # prefer explicit ID column but fall back to coord string ID
        obj_id = getattr(row, "id", None)
        if obj_id is None or str(obj_id).strip() == "":
            obj_id = coord_id
        obj_id = str(obj_id)

        try:
            _, _, sci_imas, *_ = load_imgs(tile)
        except Exception as exc:
            logging.error("Cannot load mosaics for tile %s: %s", tile, exc)
            continue

        for filt in args.filters:
            out_path = args.output_root / filt / f"{filt}_{obj_id}.jpg"
            if out_path.exists() and not args.overwrite:
                continue
            if filt not in sci_imas:
                logging.debug("Filter %s not available for tile %s", filt, tile)
                continue
            try:
                stamp = image_make_cutout(sci_imas[filt], ra, dec, args.arcsec_size, get_wcs=False)
            except Exception as exc:
                logging.warning("Cutout failed for %s %s in %s: %s", filt, obj_id, tile, exc)
                missing += 1
                continue

            if np.isnan(stamp).any() or zero_pix_fraction(stamp) >= 0.1:
                logging.debug("Skipping %s %s due to NaNs or empty borders", filt, obj_id)
                continue

            try:
                save_stamp(stamp, out_path, args.resize_px)
                success += 1
            except Exception as exc:
                logging.error("Failed to save %s: %s", out_path, exc)
                continue

    logging.info("Finished: saved %d stamps, %d failures", success, missing)


if __name__ == "__main__":
    main()
