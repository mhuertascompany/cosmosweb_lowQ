"""
Visualize confusion-matrix samples by plotting stamp cutouts for specific
true/predicted label combinations.

Usage example:

```
python -m moprhology.zoobot.plot_confusion_examples \
  --catalog /n03data/.../test_catalog.csv \
  --confusion-file /n03data/.../confusion_pairs.csv \
  --stamp-dir /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \
  --filename-template "{filter}_{id}.jpg" \
  --output /n03data/.../confusion_gallery.pdf \
  --samples-per-cell 6 \
  --rest-frame \
  --redshift-table /n07data/.../COSMOSWeb_mastercatalog_v1.fits \
  --redshift-id-column id \
  --redshift-value-column zfinal
```

The `--confusion-file` must contain columns `id_str`, `true_label`,
`pred_label` (additional columns like `filter_used` or `file_loc` are optional).
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moprhology.zoobot import train_on_cosmos_visual as cosmos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot confusion-matrix stamp samples.")
    parser.add_argument("--catalog", type=Path, required=True,
                        help="CSV with at least `id_str`, `label`, optionally `file_loc`, `filter_used`, `zfinal`.")
    parser.add_argument("--confusion-file", type=Path, required=True,
                        help="CSV listing sample IDs with `id_str`, `true_label`, `pred_label` columns.")
    parser.add_argument("--output", type=Path, default=Path("confusion_gallery.pdf"),
                        help="Destination PDF file.")
    parser.add_argument("--stamp-dir", type=Path,
                        default=Path("/n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert"),
                        help="Root directory containing stamp subfolders per filter.")
    parser.add_argument("--filter-name", default="F150W",
                        help="Filter prefix to use when not in rest-frame mode.")
    parser.add_argument("--filename-template", default="{filter}_{id}.jpg",
                        help="Template for stamp filenames, populated with filter/id.")
    parser.add_argument("--samples-per-cell", type=int, default=6,
                        help="Number of galaxies to show per (true,pred) cell.")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the grid.")
    parser.add_argument("--rest-frame", action="store_true",
                        help="Resolve stamps from rest-frame folders using filter_used or redshift.")
    parser.add_argument("--redshift-table", type=Path, default=None,
                        help="Optional redshift table for rest-frame lookup (supports CSV/TSV/Parquet/Feather/FITS).")
    parser.add_argument("--redshift-id-column", default="id",
                        help="ID column in the redshift table (rest-frame mode).")
    parser.add_argument("--redshift-value-column", default="zfinal",
                        help="Redshift value column in the redshift table (rest-frame mode).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("L"))


def resolve_stamp_path(row: pd.Series, args: argparse.Namespace, stamp_root: Path,
                       zlookup: Optional[pd.Series]) -> Path:
    if isinstance(row.get("file_loc"), str) and (not args.rest_frame):
        candidate = Path(row["file_loc"])
        if candidate.is_file():
            return candidate

    if args.rest_frame:
        filt = row.get("filter_used")
        if (not filt) and ("zfinal" in row) and pd.notna(row["zfinal"]):
            filt = cosmos.select_rest_frame_filter(row["zfinal"])
        if (not filt) and zlookup is not None:
            z_val = zlookup.get(str(row["id_str"]))
            if pd.notna(z_val):
                filt = cosmos.select_rest_frame_filter(z_val)
        if not filt:
            raise ValueError(f"Cannot determine rest-frame filter for id {row['id_str']}")
        return stamp_root / filt / args.filename_template.format(filter=filt, id=row["id_str"])

    return stamp_root / args.filename_template.format(filter=args.filter_name, id=row["id_str"])


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    rng = np.random.default_rng(args.seed)
    catalog = pd.read_csv(args.catalog)
    confusion = pd.read_csv(args.confusion_file)
    required_cols = {"id_str", "true_label", "pred_label"}
    if not required_cols.issubset(confusion.columns):
        raise ValueError(f"Confusion file must include columns: {required_cols}")

    merged = confusion.merge(catalog, on="id_str", how="left", suffixes=("", "_catalog"))
    if merged["file_loc"].isna().all():
        logging.info("file_loc column missing; will construct paths from stamp dir.")

    stamp_root = args.stamp_dir.expanduser().resolve()
    stamp_root.mkdir(parents=True, exist_ok=True)

    zlookup = None
    if args.rest_frame and "filter_used" not in merged.columns:
        zlookup = cosmos.load_redshift_lookup(args.redshift_table, args.redshift_id_column,
                                              args.redshift_value_column)

    pairs = merged.groupby(["true_label", "pred_label"])
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = args.cols
    rows_per_page = math.ceil(args.samples_per_cell / cols)

    with PdfPages(output_path) as pdf:
        for (true_lab, pred_lab), group in pairs:
            if group.empty:
                continue
            sample_n = min(len(group), args.samples_per_cell)
            sampled = group.sample(n=sample_n, random_state=rng.integers(1e9))

            fig, axes = plt.subplots(rows_per_page, cols, figsize=(3 * cols, 3 * rows_per_page))
            axes = np.atleast_2d(axes)
            flat_axes = axes.flatten()

            for ax in flat_axes:
                ax.axis("off")

            for ax, (_, row) in zip(flat_axes, sampled.iterrows()):
                try:
                    path = resolve_stamp_path(row, args, stamp_root, zlookup)
                    img = load_image(path)
                    ax.imshow(img, cmap="gray")
                    conf = row.get("pred_confidence")
                    if pd.notna(conf):
                        title = f"{row['id_str']}\nconf={conf:.2f}"
                    else:
                        title = f"{row['id_str']}"
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
                except Exception as exc:
                    ax.text(0.5, 0.5, f"Missing\n{exc}", ha="center", va="center", fontsize=7)
                    ax.set_facecolor("lightgray")
                    ax.axis("off")

            fig.suptitle(f"True: {true_lab} | Pred: {pred_lab} (n={sample_n})", fontsize=14)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    logging.info("Saved confusion gallery to %s", output_path)


if __name__ == "__main__":
    main()
