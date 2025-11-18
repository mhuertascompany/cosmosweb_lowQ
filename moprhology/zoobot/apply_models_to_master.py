"""
Apply the trained Zoobot models (regular, family, binary) to the master catalog
stamps and save all probabilities into a FITS table.

Example usage:

```
python -m moprhology.zoobot.apply_models_to_master \
  --catalog /n07data/ilbert/COSMOS-Web/photoz_MASTER_v3.1.0/MORPHO/COSMOSWeb_mastercatalog_v1.fits \
  --stamp-root /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \
  --ckpt-regular /n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/regular_6.ckpt \
  --ckpt-family /n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/family_2.ckpt \
  --ckpt-binary /n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/binary_3-v1.ckpt \
  --output ilbert_visual_zoobot_morphology.fits \
  --accelerator gpu --devices 1
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import albumentations as A
import numpy as np
import pandas as pd
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moprhology.zoobot.train_on_cosmos_visual import (
    select_rest_frame_filter,
    To3d,
)
from moprhology.zoobot import train_on_cosmos_visual as training_module
from moprhology.zoobot.generate_master_stamps import ensure_dirs  # reuse helper if needed

from zoobot.pytorch.training import finetune
from zoobot.pytorch.predictions import predict_on_catalog


LABEL_SETS: Dict[str, List[str]] = {
    "regular": [
        'ELL_REGULAR', 'ELL_INTER', 'ELL_DISTURB',
        'S0_REGULAR', 'S0_INTER', 'S0_DISTURB',
        'EDISK_REGULAR', 'EDISK_INTER', 'EDISK_DISTURB',
        'LDISK_REGULAR', 'LDISK_INTER', 'LDISK_DISTURB'
    ],
    "family": ['ELLIPTICAL', 'S0', 'EARLY_DISK', 'LATE_DISK'],
    "binary": ['NOT_DISTURBED', 'DISTURBED'],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Zoobot checkpoints to master catalog stamps.")
    parser.add_argument("--catalog", type=Path, required=True, help="Master FITS catalog.")
    parser.add_argument("--id-column", default="id", help="Column with unique IDs in the catalog.")
    parser.add_argument("--redshift-column", default="zfinal", help="Column with redshift values.")
    parser.add_argument("--stamp-root", type=Path, required=True,
                        help="Root directory containing per-filter stamp folders.")
    parser.add_argument("--filters", nargs="+", default=["F150W", "F277W", "F444W"],
                        help="Filters available in stamp-root.")
    parser.add_argument("--filename-template", default="{filter}_{id}.jpg",
                        help="How stamps are named inside each filter folder.")
    parser.add_argument("--mag-column", default="mag_model_f277w",
                        help="Optional column for magnitude filtering (case-sensitive).")
    parser.add_argument("--mag-limit", type=float, default=25.0,
                        help="Keep objects with mag_column <= mag_limit (set None to skip).")
    parser.add_argument("--ckpt-regular", type=Path, required=True, help="Checkpoint for the regular model.")
    parser.add_argument("--ckpt-family", type=Path, required=True, help="Checkpoint for the family model.")
    parser.add_argument("--ckpt-binary", type=Path, required=True, help="Checkpoint for the binary model.")
    parser.add_argument("--output", type=Path, required=True, help="Output FITS file path.")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--image-size", type=int, default=224, help="Resize dimension (square).")
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator (e.g. gpu, cpu).")
    parser.add_argument("--devices", default="auto", help="Devices argument for Lightning.")
    parser.add_argument("--precision", default="32", help="Precision for Lightning trainer.")
    parser.add_argument("--ignore-missing", action="store_true",
                        help="Skip IDs whose stamp file is missing instead of failing.")
    return parser.parse_args()


def load_master_subset(catalog_path: Path, id_col: str, z_col: str, extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
    logging.info("Reading master catalog from %s", catalog_path)

    def fetch_column(hdus, column: str) -> np.ndarray:
        for hdu in hdus[1:]:
            data = getattr(hdu, "data", None)
            if data is None or column not in data.names:
                continue
            arr = np.asarray(data[column])
            if arr.dtype.kind == "S":  # bytes -> str
                return arr.astype(str)
            if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and np.little_endian is False):
                arr = arr.byteswap().view(arr.dtype.newbyteorder('<'))
            return arr
        raise ValueError(f"Column {column} not found in FITS file {catalog_path}")

    with fits.open(catalog_path, memmap=True) as hdus:
        data = {
            id_col: fetch_column(hdus, id_col),
            z_col: fetch_column(hdus, z_col)
        }
        if extra_cols:
            for col in extra_cols:
                try:
                    data[col] = fetch_column(hdus, col)
                except ValueError:
                    logging.warning("Column %s not found; filling NaNs.", col)
                    data[col] = np.full_like(data[id_col], np.nan, dtype=float)

    return pd.DataFrame(data)


def build_inference_catalog(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stamp_root = args.stamp_root.expanduser().resolve()
    rows = []
    missing = 0
    for row in df.itertuples(index=False):
        obj_id = getattr(row, args.id_column)
        z = getattr(row, args.redshift_column)
        if pd.isna(z):
            continue
        filter_name = select_rest_frame_filter(float(z))
        if filter_name is None:
            continue
        file_path = stamp_root / filter_name / args.filename_template.format(filter=filter_name, id=obj_id)
        if not file_path.exists():
            missing += 1
            continue
        rows.append({
            'id': str(obj_id),
            'file_loc': str(file_path),
            'filter_used': filter_name,
            args.redshift_column: z
        })
    logging.info("Prepared %d entries for inference (skipped %d missing files).", len(rows), missing)
    return pd.DataFrame(rows)



def get_inference_transform(image_size: int):
    import torchvision.transforms as T
    return T.Compose([
        T.Grayscale(3),
        T.Resize((image_size, image_size)),
        T.ConvertImageDtype(torch.float32)
    ])


def run_model(model_path: Path, label_names: List[str], catalog: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    logging.info("Loading model from %s", model_path)
    model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(str(model_path), strict=False)
    transform = get_inference_transform(args.image_size)

    catalog = catalog.rename(columns={'id': 'id_str'})
    kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers
    }
    if transform is not None:
        kwargs['test_transform'] = transform
    preds = predict_on_catalog.predict(
        catalog,
        model,
        label_cols=label_names,
        inference_transform=None,
        save_loc=None,
        datamodule_kwargs=kwargs,
        trainer_kwargs={
            'accelerator': args.accelerator,
            'devices': args.devices,
            'precision': args.precision
        }
    )
    return preds


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    extra_cols = [args.mag_column] if (args.mag_column and args.mag_limit is not None) else []
    df_master = load_master_subset(args.catalog, args.id_column, args.redshift_column, extra_cols)
    if args.mag_column and args.mag_column in df_master.columns and args.mag_limit is not None:
         before = len(df_master)
         df_master = df_master[df_master[args.mag_column].notna() & (df_master[args.mag_column] <= args.mag_limit)]
         logging.info("Magnitude cut %s <= %.2f reduced catalog from %d to %d objects.",
                      args.mag_column, args.mag_limit, before, len(df_master))
    inference_catalog = build_inference_catalog(df_master, args)

    outputs = {'id': inference_catalog['id'].astype(str)}

    model_configs = [
        ('regular', args.ckpt_regular),
        ('family', args.ckpt_family),
        ('binary', args.ckpt_binary),
    ]

    for label_set, ckpt in model_configs:
        label_names = LABEL_SETS[label_set]
        logging.info("Running %s model (%d classes)", label_set, len(label_names))
        preds = run_model(ckpt, label_names, inference_catalog[['id', 'file_loc']], args)
        preds = preds.rename(columns={name: f"{label_set}_{name}" for name in label_names})
        outputs.update({col: preds[col].values for col in preds.columns if col not in {'id'}})

    final_df = pd.DataFrame(outputs)
    final_df['id'] = final_df['id'].astype(str)

    table = Table.from_pandas(final_df)
    table.write(args.output, overwrite=True)
    logging.info("Saved predictions to %s", args.output)


if __name__ == "__main__":
    main()
