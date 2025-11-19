"""
Create a PDF gallery of random COSMOS-Web stamps drawn from the final prediction catalog.

Example:

python -m moprhology.zoobot.plot_prediction_examples \
  --predictions /n03data/.../ilbert_visual_zoobot_morphology.fits \
  --master-catalog /n07data/.../COSMOSWeb_mastercatalog_v1.fits \
  --stamp-root /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \
  --label-set regular \
  --samples-per-class 8 \
  --output cosmos_regular_examples.pdf
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image
from astropy.table import Table

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moprhology.zoobot.train_on_cosmos_visual import select_rest_frame_filter
from moprhology.zoobot.apply_models_to_master import (
    load_master_subset,
    sanitize_id,
)


LABEL_SETS: Dict[str, Sequence[str]] = {
    "regular": ["ELL", "S0", "EARLY_DISK", "LATE_DISK"],
    "family": ["ELLIPTICAL", "S0", "EARLY_DISK", "LATE_DISK"],
    "binary": ["NOT_DISTURBED", "DISTURBED"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot random examples from the final prediction catalog.")
    parser.add_argument("--predictions", type=Path, required=True, help="FITS/CSV file with prediction columns.")
    parser.add_argument("--master-catalog", type=Path, required=True, help="Master FITS catalog used to build stamps.")
    parser.add_argument("--stamp-root", type=Path, required=True, help="Directory containing per-filter stamp folders.")
    parser.add_argument("--filename-template", default="{filter}_{id}.jpg", help="Filename pattern inside each filter dir.")
    parser.add_argument("--label-set", choices=list(LABEL_SETS.keys()), default="regular", help="Which label set to visualize.")
    parser.add_argument("--samples-per-class", type=int, default=6, help="Number of random objects per class.")
    parser.add_argument("--output", type=Path, default=Path("prediction_examples.pdf"), help="Destination PDF path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--id-column", default="id", help="ID column name in master catalog.")
    parser.add_argument("--redshift-column", default="zfinal", help="Redshift column name in master catalog.")
    parser.add_argument("--mag-column", default=None, help="Optional magnitude column for filtering.")
    parser.add_argument("--mag-limit", type=float, default=None, help="If set, keep rows with mag <= limit when building lookup.")
    return parser.parse_args()


def load_predictions_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fz"}:
        table = Table.read(path)
        df = table.to_pandas()
    else:
        df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError("Prediction file must contain an 'id' column.")
    df["id"] = df["id"].astype(str)
    return df


def build_file_lookup(
    master_catalog: Path,
    id_column: str,
    redshift_column: str,
    needed_ids: set[str],
    stamp_root: Path,
    filename_template: str,
    mag_column: str | None,
    mag_limit: float | None,
) -> Dict[str, Path]:
    extra_cols = [c for c in [mag_column] if c]
    df_master = load_master_subset(master_catalog, id_column, redshift_column, extra_cols)
    df_master[id_column] = df_master[id_column].apply(sanitize_id).astype(str)
    subset = df_master[df_master[id_column].isin(needed_ids)].copy()

    if mag_column and mag_column in subset.columns and mag_limit is not None:
        subset = subset[subset[mag_column].notna() & (subset[mag_column] <= mag_limit)]

    lookup: Dict[str, Path] = {}
    missing = 0
    for row in subset.itertuples(index=False):
        obj_id = getattr(row, id_column)
        z = getattr(row, redshift_column)
        if pd.isna(z):
            continue
        filt = select_rest_frame_filter(float(z))
        if filt is None:
            continue
        path = stamp_root / filt / filename_template.format(filter=filt, id=obj_id)
        if path.exists():
            lookup[str(obj_id)] = path
        else:
            missing += 1
    if missing:
        logging.warning("Missing %d stamp files referenced by master catalog.", missing)
    return lookup


def choose_examples(
    predictions: pd.DataFrame,
    class_columns: List[str],
    samples_per_class: int,
    rng: random.Random,
) -> Dict[str, pd.DataFrame]:
    preds = predictions.copy()
    preds["pred_class"] = preds[class_columns].idxmax(axis=1)
    selections: Dict[str, pd.DataFrame] = {}
    for col in class_columns:
        candidates = preds[preds["pred_class"] == col]
        if candidates.empty:
            selections[col] = candidates
            continue
        n = min(samples_per_class, len(candidates))
        selections[col] = candidates.sample(n=n, random_state=rng.randint(0, 10_000))
    return selections


def plot_examples(
    selections: Dict[str, pd.DataFrame],
    class_names: Dict[str, str],
    output: Path,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for column, df in selections.items():
            display_name = class_names.get(column, column)
            if df.empty:
                logging.warning("No examples found for %s", display_name)
                continue
            fig, axes = plt.subplots(1, len(df), figsize=(3 * len(df), 3))
            if len(df) == 1:
                axes = [axes]
            for ax, (_, row) in zip(axes, df.iterrows()):
                loc = row.get("file_loc")
                try:
                    image = Image.open(loc).convert("L")
                    ax.imshow(np.array(image), cmap="gray")
                    prob = row[column]
                    ax.set_title(f"{row['id']}\nP={prob:.2f}")
                except Exception as exc:
                    ax.text(0.5, 0.5, f"Missing\n{exc}", ha="center", va="center")
                    ax.set_facecolor("lightgray")
                ax.axis("off")
            fig.suptitle(display_name, fontsize=14)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    logging.info("Saved gallery to %s", output)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    predictions = load_predictions_table(args.predictions)
    label_suffixes = LABEL_SETS[args.label_set]
    prefix = f"{args.label_set}_"
    class_columns = [f"{prefix}{name}" for name in label_suffixes if f"{prefix}{name}" in predictions.columns]
    if not class_columns:
        raise ValueError(f"No columns starting with '{prefix}' found in {args.predictions}")

    stamp_root = args.stamp_root.expanduser().resolve()
    needed_ids = set(predictions["id"].astype(str))
    file_lookup = build_file_lookup(
        args.master_catalog,
        args.id_column,
        args.redshift_column,
        needed_ids,
        stamp_root,
        args.filename_template,
        args.mag_column,
        args.mag_limit,
    )

    predictions["file_loc"] = predictions["id"].map(file_lookup)
    missing = predictions["file_loc"].isna().sum()
    if missing:
        logging.warning("Missing stamp paths for %d/%d objects.", missing, len(predictions))
    available = predictions.dropna(subset=["file_loc"])
    if available.empty:
        raise RuntimeError("No stamp paths resolved; cannot plot examples.")

    rng = random.Random(args.seed)
    selections = choose_examples(available, class_columns, args.samples_per_class, rng)
    class_name_map = {f"{prefix}{suffix}": suffix.replace("_", " ") for suffix in label_suffixes}
    output = args.output.expanduser()
    plot_examples(selections, class_name_map, output)


if __name__ == "__main__":
    main()
