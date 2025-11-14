"""
Create a PDF gallery of random REGULAR-class COSMOS-Web stamps to visually inspect label fidelity.

Example usage:

```
python -m moprhology.zoobot.plot_regular_examples \
  --catalog /Users/.../test_catalog.csv \
  --predictions /Users/.../test_set_regular.csv \
  --stamp-dir /n03data/huertas/COSMOS-Web/zoobot/stamps/f150w \
  --filter-name F150W \
  --filename-template "{filter}_{id}.jpg" \
  --samples-per-class 6 \
  --output examples_regular.pdf
```
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

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
    parser = argparse.ArgumentParser(description="Plot random REGULAR stamp examples by class.")
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="CSV containing at least columns 'id_str', 'label', and optionally 'file_loc'.",
    )
    parser.add_argument(
        "--stamp-dir",
        type=Path,
        default=Path("/n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert"),
        help="Root directory with stamp JPEGs. For rest-frame mode, expect per-filter subfolders.",
    )
    parser.add_argument(
        "--filter-name",
        default="F150W",
        help="Filter prefix used in stamp filenames (ignored in rest-frame mode).",
    )
    parser.add_argument(
        "--filename-template",
        default="{filter}_{id}.jpg",
        help="Template for locating stamps when file_loc missing. Can reference {filter} and {id}.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=6,
        help="Number of random examples to draw for each REGULAR class.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("regular_examples.pdf"),
        help="Destination PDF path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--rest-frame",
        action="store_true",
        help="Use rest-frame stamps stored in per-filter folders under --stamp-dir.",
    )
    parser.add_argument(
        "--redshift-table",
        type=Path,
        default=None,
        help="CSV/TSV/Parquet table containing 'id_str' and 'zfinal' columns (only needed for rest-frame mode).",
    )
    parser.add_argument(
        "--redshift-value-column",
        default="zfinal",
        help="Column containing redshift values when --rest-frame is enabled.",
    )
    return parser.parse_args()


def resolve_file_loc(
    row: pd.Series,
    stamp_dir: Path,
    filename_template: str,
    filter_name: str,
    rest_frame: bool,
) -> Path:
    if "file_loc" in row and isinstance(row["file_loc"], str) and not rest_frame:
        path = Path(row["file_loc"])
        if path.is_file():
            return path

    if rest_frame:
        if "filter_used" in row and isinstance(row["filter_used"], str):
            filt = row["filter_used"]
        elif "zfinal" in row:
            filt = cosmos.select_rest_frame_filter(row["zfinal"])
        else:
            raise ValueError("Rest-frame mode requires 'filter_used' or 'zfinal' in the catalog.")
        if filt is None:
            raise ValueError("Unable to determine rest-frame filter for row %s" % row)
        return stamp_dir / filt / filename_template.format(filter=filt, id=row["id_str"])

    return stamp_dir / filename_template.format(filter=filter_name, id=row["id_str"])


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img.convert("L"))
    return arr


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cosmos.set_label_columns("regular")
    label_names = cosmos.LABEL_COLUMNS.copy()
    _, id_to_class = cosmos.label_name_mappings()

    catalog = pd.read_csv(args.catalog)
    if "label" not in catalog.columns or "id_str" not in catalog.columns:
        raise ValueError("Catalog must contain 'id_str' and 'label' columns.")

    stamp_dir = args.stamp_dir.expanduser().resolve()
    stamp_dir.mkdir(parents=True, exist_ok=True)

    if args.rest_frame:
        if "filter_used" not in catalog.columns and args.redshift_table is None:
            raise ValueError("Rest-frame mode requires 'filter_used' in catalog or --redshift-table with redshifts.")
        if "filter_used" not in catalog.columns:
            ztable = pd.read_csv(args.redshift_table)
            if "id_str" not in ztable.columns or args.redshift_value_column not in ztable.columns:
                raise ValueError("Redshift table must contain 'id_str' and the specified z column.")
            zlookup = ztable.set_index('id_str')[args.redshift_value_column]
            catalog['zfinal'] = catalog['id_str'].map(zlookup)

    rng = random.Random(args.seed)

    grouped: Dict[int, pd.DataFrame] = {
        idx: catalog[catalog["label"] == idx] for idx in range(len(label_names))
    }

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for idx, df in grouped.items():
            class_name = id_to_class.get(idx, f"class_{idx}")
            if df.empty:
                logging.warning("No samples found for class %s", class_name)
                continue

            count = min(args.samples_per_class, len(df))
            selected = df.sample(n=count, random_state=rng.randint(0, 10_000))

            fig, axes = plt.subplots(1, count, figsize=(3 * count, 3))
            if count == 1:
                axes = [axes]

            for ax, (_, row) in zip(axes, selected.iterrows()):
                try:
                    path = resolve_file_loc(
                        row=row,
                        stamp_dir=stamp_dir,
                        filename_template=args.filename_template,
                        filter_name=args.filter_name,
                        rest_frame=args.rest_frame,
                    )
                    image = load_image(path)
                    ax.imshow(image, cmap="gray")
                    ax.set_title(f"{row['id_str']}")
                except Exception as exc:
                    ax.text(0.5, 0.5, f"Missing\n{exc}", ha="center", va="center")
                    ax.set_facecolor("lightgray")
                ax.axis("off")

            fig.suptitle(f"{class_name} (n={count})", fontsize=14)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    logging.info("Saved gallery to %s", output_path)


if __name__ == "__main__":
    main()
