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

from . import train_on_cosmos_visual as cosmos


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
        default=None,
        help="Directory with stamp JPEGs. Required if catalog lacks 'file_loc'.",
    )
    parser.add_argument(
        "--filter-name",
        default="F150W",
        help="Filter prefix used in stamp filenames (default: %(default)s).",
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
    return parser.parse_args()


def resolve_file_loc(
    row: pd.Series,
    stamp_dir: Optional[Path],
    filename_template: str,
    filter_name: str,
) -> Path:
    if "file_loc" in row and isinstance(row["file_loc"], str):
        path = Path(row["file_loc"])
        if path.is_file():
            return path
    if stamp_dir is None:
        raise FileNotFoundError(
            "Catalog row lacks 'file_loc' and --stamp-dir was not provided."
        )
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

    stamp_dir = args.stamp_dir.expanduser().resolve() if args.stamp_dir else None
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

