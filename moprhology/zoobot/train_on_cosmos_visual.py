"""
Finetune a Zoobot encoder on COSMOS-Web visual morphology labels.

This script mirrors ``train_on_gz_ceers_tree.py`` but targets the discrete
visual classes stored in the COSMOS visual annotation database described in
``moprhology/read_catalogues_v7.py``. It builds a catalog that links each
galaxy id to its stamp on disk, splits the data, defines augmentations, and
finetunes a ``FinetuneableZoobotClassifier`` seeded from the Euclid encoder
published on Hugging Face.

Example usage (run from repo root):

```
python -m moprhology.zoobot.train_on_cosmos_visual \
  --stamp-dir /n03data/huertas/COSMOS-Web/zoobot/stamps/f150w \
  --visual-labels /n07data/ilbert/COSMOS-Web/photoz_MASTER_v3.1.0/MORPHO/visualmorpho_COSMOSWeb_v7.db \
  --sqlite-table morphology \
  --filter-name F150W \
  --save-dir /n03data/huertas/COSMOS-Web/zoobot/models/visual_finetune
```
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
import math
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule as GalaxyDataModule
from galaxy_datasets.transforms import get_galaxy_transform, minimal_view_config

from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.pytorch.training import finetune

# Visual morphology labels of interest
CLASS_COLUMNS: List[str] = [
    'ELL_REGULAR', 'ELL_INTER', 'ELL_DISTURB',
    'S0_REGULAR', 'S0_INTER', 'S0_DISTURB',
    'EDISK_REGULAR', 'EDISK_INTER', 'EDISK_DISTURB',
    'LDISK_REGULAR', 'LDISK_INTER', 'LDISK_DISTURB'
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_COLUMNS)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}


class To3d:
    """Albumentations-compatible helper to triplicate a greyscale cutout."""

    def __call__(self, image, **kwargs):
        if image.ndim == 2:
            return np.stack((image, image, image), axis=-1)
        if image.ndim == 3 and image.shape[-1] == 1:
            return np.repeat(image, 3, axis=-1)
        return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune Zoobot on COSMOS visual labels.")
    parser.add_argument('--stamp-dir', required=True, type=Path,
                        help="Directory containing COSMOS cutouts (jpg).")
    parser.add_argument('--filter-name', default='F150W',
                        help="Filter prefix used in the stamp filenames (default: %(default)s).")
    parser.add_argument('--filename-template', default='{filter}_{id}.jpg',
                        help="Template used to locate each stamp. "
                             "Formatted with keys 'filter' and 'id'.")
    parser.add_argument('--visual-labels', type=Path, default=None,
                        help="Optional CSV/Parquet/SQLite file with the visual morphology table. "
                             "If omitted, the script attempts to import moprhology.read_catalogues_v7.")
    parser.add_argument('--sqlite-table', default='morphology',
                        help="Table name inside the SQLite DB (when --visual-labels points to .db/.sqlite).")
    parser.add_argument('--max-galaxies', type=int, default=None,
                        help="Optional cap on the number of galaxies for quick experiments.")
    parser.add_argument('--test-fraction', type=float, default=0.2,
                        help="Fraction of galaxies reserved for the held-out test split.")
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help="Fraction of galaxies reserved for validation.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for all splits.")
    parser.add_argument('--batch-size', type=int, default=64, help="Training batch size.")
    parser.add_argument('--num-workers', type=int, default=8, help="DataLoader worker count.")
    parser.add_argument('--image-size', type=int, default=224,
                        help="Side length (pixels) after cropping/resizing.")
    parser.add_argument('--crop-scale', type=float, nargs=2, default=(0.7, 0.8),
                        help="Scale bounds for RandomResizedCrop.")
    parser.add_argument('--crop-ratio', type=float, nargs=2, default=(0.9, 1.1),
                        help="Aspect ratio bounds for RandomResizedCrop.")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument('--weight-decay', type=float, default=0.05, help="AdamW weight decay.")
    parser.add_argument('--layer-decay', type=float, default=0.75,
                        help="Layer-wise LR decay applied inside Zoobot.")
    parser.add_argument('--head-dropout', type=float, default=0.5,
                        help="Dropout probability inside the classification head.")
    parser.add_argument('--training-mode', choices=['full', 'head_only'], default='full',
                        help="Whether to finetune the full encoder or only the linear head.")
    parser.add_argument('--encoder-name', default='hf_hub:mwalmsley/zoobot-encoder-euclid',
                        help="Zoobot Hugging Face encoder checkpoint to start from.")
    parser.add_argument('--max-epochs', type=int, default=50, help="Maximum finetuning epochs.")
    parser.add_argument('--patience', type=int, default=8,
                        help="Early stopping patience (epochs).")
    parser.add_argument('--accelerator', default='auto',
                        help="PyTorch Lightning accelerator argument (e.g. 'gpu', 'cpu').")
    parser.add_argument('--devices', default='auto',
                        help="Devices argument passed to Lightning (e.g. 1, [0,1], 'auto').")
    parser.add_argument('--precision', default='32', help="Precision argument for Lightning Trainer.")
    parser.add_argument('--save-dir', type=Path, default=Path('./zoobot_cosmos_visual'),
                        help="Directory where checkpoints and logs will be written.")
    parser.add_argument('--prediction-csv', type=Path, default=None,
                        help="Optional CSV to store per-class softmax predictions on the test split.")
    parser.add_argument('--n-predict-samples', type=int, default=1,
                        help="If supported, number of stochastic forward passes for predictions.")
    parser.add_argument('--use-class-weights', action='store_true',
                        help="Enable balanced class weights inside the cross-entropy loss.")
    parser.add_argument('--keep-ambiguous', action='store_true',
                        help="Keep galaxies with multiple positive labels (defaults to dropping them).")
    parser.add_argument('--disable-progbar', action='store_true',
                        help="Hide Lightning progress bars.")
    return parser.parse_args()


def load_visual_catalog(path: Optional[Path], table: str) -> pd.DataFrame:
    """Load the visual morphology table."""
    if path is None:
        logging.info("Loading visual morphology table via moprhology.read_catalogues_v7.read_visual_morpho()")
        try:
            from moprhology import read_catalogues_v7
        except ImportError as exc:
            raise RuntimeError(
                "read_catalogues_v7 is unavailable. Please pass --visual-labels pointing to the visual database."
            ) from exc
        df_visual = read_catalogues_v7.read_visual_morpho()
    else:
        logging.info("Loading visual morphology table from %s", path)
        suffix = path.suffix.lower()
        if suffix in {'.csv', '.txt'}:
            df_visual = pd.read_csv(path)
        elif suffix in {'.tsv'}:
            df_visual = pd.read_csv(path, sep='\t')
        elif suffix in {'.parquet'}:
            df_visual = pd.read_parquet(path)
        elif suffix in {'.feather'}:
            df_visual = pd.read_feather(path)
        elif suffix in {'.db', '.sqlite'}:
            query = f"SELECT id, {', '.join(CLASS_COLUMNS)} FROM {table}"
            with sqlite3.connect(path) as conn:
                df_visual = pd.read_sql_query(query, conn)
        else:
            raise ValueError(f"Unsupported label file type: {suffix}")
    missing = [col for col in ['id', *CLASS_COLUMNS] if col not in df_visual.columns]
    if missing:
        raise ValueError(f"Missing required columns in visual table: {missing}")
    return df_visual[['id', *CLASS_COLUMNS]].copy()


def attach_stamps(
    df_visual: pd.DataFrame,
    stamp_dir: Path,
    filename_template: str,
    filter_name: str,
    keep_ambiguous: bool
) -> pd.DataFrame:
    """Build catalog rows that include the stamp path and encoded label."""
    stamp_dir = stamp_dir.expanduser().resolve()
    if not stamp_dir.exists():
        raise FileNotFoundError(f"Stamp directory {stamp_dir} does not exist.")

    def sanitize_id(raw_id):
        if isinstance(raw_id, (int, np.integer)):
            return int(raw_id)
        if isinstance(raw_id, (float, np.floating)):
            if math.isfinite(raw_id):
                rounded = round(float(raw_id))
                if abs(float(raw_id) - rounded) < 1e-6:
                    return int(rounded)
        if isinstance(raw_id, str):
            try:
                return int(raw_id)
            except ValueError:
                return raw_id
        return raw_id

    preview = [sanitize_id(pid) for pid in df_visual['id'].head(10).tolist()]
    preview_paths = [
        stamp_dir / filename_template.format(filter=filter_name, id=pid)
        for pid in preview
    ]
    logging.info(
        "First visual IDs: %s",
        ", ".join(str(pid) for pid in preview)
    )
    logging.info(
        "Corresponding expected stamp files: %s",
        ", ".join(str(path) for path in preview_paths)
    )

    rows = []
    missing_files = 0
    ambiguous = 0
    unlabeled = 0

    for record in df_visual.itertuples(index=False):
        clean_id = sanitize_id(record.id)
        file_loc = stamp_dir / filename_template.format(filter=filter_name, id=clean_id)
        if not file_loc.exists():
            missing_files += 1
            continue
        labels = np.array([getattr(record, col) for col in CLASS_COLUMNS], dtype=float)
        labels = np.nan_to_num(labels, nan=0.0)
        if labels.sum() == 0:
            unlabeled += 1
            continue

        positives = np.flatnonzero(labels >= 0.5)
        if len(positives) > 1 and not keep_ambiguous:
            ambiguous += 1
            continue
        class_idx = int(positives[0] if len(positives) else np.argmax(labels))
        rows.append({
            'id_str': str(clean_id),
            'file_loc': str(file_loc),
            'label': class_idx,
            **{col: getattr(record, col) for col in CLASS_COLUMNS}
        })

    logging.info(
        "Catalog built with %d usable galaxies (%d missing stamps, %d unlabeled, %d ambiguous dropped)",
        len(rows), missing_files, unlabeled, ambiguous
    )
    if not rows:
        raise RuntimeError("No usable galaxies found after filtering -- aborting.")
    return pd.DataFrame(rows)


def maybe_subsample(catalog: pd.DataFrame, max_galaxies: Optional[int], seed: int) -> pd.DataFrame:
    if max_galaxies is None or max_galaxies >= len(catalog):
        return catalog
    logging.info("Subsampling %d galaxies out of %d total for quick experiment", max_galaxies, len(catalog))
    return catalog.sample(n=max_galaxies, random_state=seed).reset_index(drop=True)


def stratified_splits(
    catalog: pd.DataFrame,
    test_fraction: float,
    val_fraction: float,
    seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_fraction <= 0 or val_fraction <= 0 or test_fraction + val_fraction >= 1.0:
        raise ValueError("Require 0 < val_fraction, test_fraction and val+test < 1.")

    train_val, test = train_test_split(
        catalog,
        test_size=test_fraction,
        stratify=catalog['label'],
        random_state=seed
    )
    rel_val_fraction = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=rel_val_fraction,
        stratify=train_val['label'],
        random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_train_transforms(image_size: int, crop_scale: Iterable[float], crop_ratio: Iterable[float]) -> A.Compose:
    return A.Compose([
        A.Lambda(image=To3d(), always_apply=True),
        A.Rotate(limit=180, interpolation=1, border_mode=0, value=0, always_apply=True),
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=tuple(crop_scale),
            ratio=tuple(crop_ratio),
            interpolation=1,
            always_apply=True
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])


def build_eval_transform(image_size: int):
    cfg = minimal_view_config()
    cfg.output_size = image_size
    cfg.random_affine = dict(degrees=0, translate=None, scale=(1.0, 1.0), shear=None)
    cfg.center_crop = True
    cfg.random_resized_crop = False
    cfg.flip_prob = 0.0
    return get_galaxy_transform(cfg)


def compute_weights(train_catalog: pd.DataFrame) -> np.ndarray:
    classes = np.array(sorted(ID_TO_CLASS.keys()))
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_catalog['label'].values
    )
    logging.info("Class weights: %s", dict(zip(classes, weights)))
    return weights


def create_datamodule(
    train_catalog: pd.DataFrame,
    val_catalog: pd.DataFrame,
    test_catalog: pd.DataFrame,
    train_transform: A.Compose,
    batch_size: int,
    num_workers: int
) -> GalaxyDataModule:
    kwargs = dict(
        label_cols=['label'],
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        batch_size=batch_size,
        num_workers=num_workers
    )
    try:
        kwargs['custom_albumentation_transform'] = train_transform
        return GalaxyDataModule(**kwargs)
    except TypeError:
        logging.warning("Installed galaxy-datasets version does not accept custom_albumentation_transform; falling back to defaults.")
        kwargs.pop('custom_albumentation_transform', None)
        return GalaxyDataModule(**kwargs)


def require_gpu(accelerator: str) -> None:
    """Ensure training runs on GPU to match cluster expectations."""
    if accelerator not in {'gpu', 'cuda'}:
        raise RuntimeError(
            f"GPU training required, but --accelerator is set to '{accelerator}'. "
            "Pass --accelerator gpu (or cuda) to use available devices."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU accelerator requested but torch.cuda reports no available CUDA devices."
        )
    logging.info("CUDA detected: %d device(s) visible.", torch.cuda.device_count())


def finetune_model(args: argparse.Namespace) -> tuple[finetune.FinetuneableZoobotClassifier, GalaxyDataModule, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_visual = load_visual_catalog(args.visual_labels, args.sqlite_table)
    catalog = attach_stamps(df_visual, args.stamp_dir, args.filename_template, args.filter_name, args.keep_ambiguous)
    catalog = maybe_subsample(catalog, args.max_galaxies, args.seed)

    train_catalog, val_catalog, test_catalog = stratified_splits(
        catalog,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed
    )

    logging.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(train_catalog), len(val_catalog), len(test_catalog)
    )

    class_weights = compute_weights(train_catalog) if args.use_class_weights else None

    train_transform = build_train_transforms(args.image_size, args.crop_scale, args.crop_ratio)
    datamodule = create_datamodule(
        train_catalog,
        val_catalog,
        test_catalog,
        train_transform,
        args.batch_size,
        args.num_workers
    )

    model = finetune.FinetuneableZoobotClassifier(
        name=args.encoder_name,
        num_classes=len(CLASS_COLUMNS),
        label_col='label',
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,
        head_dropout_prob=args.head_dropout,
        training_mode=args.training_mode,
        class_weights=class_weights,
        prog_bar=not args.disable_progbar
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    require_gpu(args.accelerator)

    trainer = finetune.get_trainer(
        save_dir=str(args.save_dir),
        max_epochs=args.max_epochs,
        patience=args.patience,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision
    )

    logging.info("Starting finetuning with encoder %s", args.encoder_name)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    return model, datamodule, train_catalog, val_catalog, test_catalog


def run_predictions(
    args: argparse.Namespace,
    model,
    test_catalog: pd.DataFrame
) -> None:
    if args.prediction_csv is None:
        return

    prediction_cols = [f"p_{name}" for name in CLASS_COLUMNS]
    eval_transform = build_eval_transform(args.image_size)

    predict_kwargs = dict(
        catalog=test_catalog,
        model=model,
        label_cols=prediction_cols,
        inference_transform=eval_transform,
        save_loc=str(args.prediction_csv),
        datamodule_kwargs={'batch_size': args.batch_size, 'num_workers': args.num_workers}
    )

    try:
        prediction_fn = predict_on_catalog.predict  # type: ignore[attr-defined]
        prediction_fn(n_samples=args.n_predict_samples, **predict_kwargs)
    except TypeError:
        logging.warning("predict_on_catalog.predict signature does not support n_samples; running single deterministic pass.")
        predict_on_catalog.predict(**{k: v for k, v in predict_kwargs.items() if k != 'n_samples'})


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    L.seed_everything(args.seed, workers=True)

    model, datamodule, _, _, test_catalog = finetune_model(args)
    run_predictions(args, model, test_catalog)


if __name__ == '__main__':
    main()
