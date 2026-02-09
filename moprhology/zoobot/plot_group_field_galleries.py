"""
Create PDF galleries of stamps for quenched galaxies in groups and field,
across mass bins and family morphologies. Adds redshift and mass labels.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from PIL import Image

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moprhology.zoobot.train_on_cosmos_visual import select_rest_frame_filter


FAMILY_COLS = ['family_ELLIPTICAL', 'family_S0', 'family_EARLY_DISK', 'family_LATE_DISK']
FAMILY_LABELS = ['elliptical', 's0', 'early-disk', 'late-disk']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stamp galleries for group/field quenched galaxies.")
    parser.add_argument("--catalog", type=Path, required=True, help="Master FITS catalog.")
    parser.add_argument("--morphology", type=Path, required=True, help="Morphology FITS with family columns.")
    parser.add_argument("--group-dir", type=Path, required=True, help="Directory with groups.fits/memberships.fits.")
    parser.add_argument("--stamp-root", type=Path, required=True, help="Root directory with stamp folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save PDF galleries.")
    parser.add_argument("--samples-per-class", type=int, default=12, help="Number of stamps per class.")
    parser.add_argument("--mass-bins", default="8-9,9-10,10-10.5",
                        help="Mass bins as comma-separated ranges, e.g. 8-9,9-10.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def to_native(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.byteorder == '>' or (arr.dtype.byteorder == '=' and np.little_endian is False):
        arr = arr.byteswap().newbyteorder()
    return arr


def normalize_ids(values) -> pd.Series:
    series = pd.Series(values).astype(str).str.strip()
    series = series.str.replace(r'\.0$', '', regex=True)
    return series


def read_master_catalog(path: Path) -> pd.DataFrame:
    photom_cols = ['id', 'warn_flag', 'mag_model_f444w', 'flag_star_hsc', 'ra_model', 'dec_model']
    lephare_cols = ['type', 'zfinal', 'mass_med', 'ssfr_med', 'mabs_nuv', 'mabs_r', 'mabs_j']

    photom = Table.read(path, hdu=1)[photom_cols]
    lephare = Table.read(path, hdu=2)[lephare_cols]

    data = {}
    for col in lephare_cols:
        data[col] = to_native(lephare[col])
    for col in photom_cols:
        data[col] = to_native(photom[col])

    df = pd.DataFrame(data)
    return df


def load_family_morphology(path: Path, catalog_ids: pd.Series) -> np.ndarray:
    morph = Table.read(path)
    morph_df = morph.to_pandas()
    morph_df['id'] = normalize_ids(morph_df['id'])

    matrix = (
        morph_df.set_index('id')
        .reindex(catalog_ids)[FAMILY_COLS]
        .to_numpy()
    )
    return matrix


def build_clean_mask(catalog: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mabs_nuv = np.ma.filled(np.asarray(catalog['mabs_nuv'], dtype=float), np.nan)
    mabs_r = np.ma.filled(np.asarray(catalog['mabs_r'], dtype=float), np.nan)
    mabs_j = np.ma.filled(np.asarray(catalog['mabs_j'], dtype=float), np.nan)
    mass_log = np.ma.filled(np.asarray(catalog['mass_med'], dtype=float), np.nan)
    z = np.asarray(catalog['zfinal'], dtype=float)

    nuv_minus_r = mabs_nuv - mabs_r
    r_minus_j = mabs_r - mabs_j

    catalog['nuv_minus_r'] = nuv_minus_r
    catalog['r_minus_j'] = r_minus_j

    clean_mask = (
        (np.asarray(catalog['type']) == 0) &
        (np.asarray(catalog['warn_flag']) == 0) &
        (np.abs(np.asarray(catalog['mag_model_f444w'])) < 30) &
        (np.asarray(catalog['flag_star_hsc']) == 0)
    )

    finite_mask = np.isfinite(nuv_minus_r) & np.isfinite(r_minus_j) & np.isfinite(mass_log)
    clean_mask &= finite_mask

    low_mass_mask = mass_log < 10.0
    quiescent_mask = (nuv_minus_r > 3.1) & (nuv_minus_r > 3.0 * r_minus_j + 1.0)

    final_mask = clean_mask & low_mass_mask & quiescent_mask
    return final_mask, mass_log, z, nuv_minus_r


def load_group_membership(group_dir: Path, catalog_ids: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    groups_path = group_dir / 'groups.fits'
    member_path = group_dir / 'memberships.fits'

    groups = Table.read(groups_path).to_pandas()
    member = Table.read(member_path).to_pandas()

    groups = groups[
        (groups['SN_NOCL'] >= 10) &
        (groups['LAMBDA_STAR'] > 10.55) &
        (groups['MSKFRC'] < 0.2)
    ]

    member = member[member['ID'].isin(groups['ID'])]

    member['GALID'] = member['GALID'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    field_filtered = member[member['FIELD_PROB'] > 0.8]
    idx_field = field_filtered.groupby('GALID')['ASSOC_PROB'].idxmin()
    field_unique = field_filtered.loc[idx_field]
    field_ids = set(field_unique['GALID'].astype(str))

    group_filtered = member[member['ASSOC_PROB'] > 0.5]
    idx_group = group_filtered.groupby('GALID')['ASSOC_PROB'].idxmax()
    group_unique = group_filtered.loc[idx_group]
    group_ids = set(group_unique['GALID'].astype(str))

    field_mask = catalog_ids.isin(field_ids).to_numpy()
    group_mask = catalog_ids.isin(group_ids).to_numpy()

    return group_mask, field_mask, groups, member


def build_matched_masks(
    base_mask: np.ndarray,
    mass_log: np.ndarray,
    group_mask: np.ndarray,
    field_mask: np.ndarray,
    bins: List[Tuple[float, float]],
    rng: np.random.Generator
) -> Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    matched: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
    for lo, hi in bins:
        bin_mask = base_mask & (mass_log >= lo) & (mass_log < hi)
        group_ids = np.where(bin_mask & group_mask)[0]
        field_ids = np.where(bin_mask & field_mask)[0]

        n = min(len(group_ids), len(field_ids))
        if n == 0:
            matched[(lo, hi)] = (np.zeros(len(base_mask), dtype=bool),
                                 np.zeros(len(base_mask), dtype=bool))
            continue

        group_sel = rng.choice(group_ids, size=n, replace=False)
        field_sel = rng.choice(field_ids, size=n, replace=False)

        group_mask_matched = np.zeros(len(base_mask), dtype=bool)
        field_mask_matched = np.zeros(len(base_mask), dtype=bool)
        group_mask_matched[group_sel] = True
        field_mask_matched[field_sel] = True

        matched[(lo, hi)] = (group_mask_matched, field_mask_matched)
    return matched


def select_stamp_path(stamp_root: Path, obj_id: str, z: float, filename_template: str) -> Path | None:
    filt = select_rest_frame_filter(float(z))
    if filt is None:
        return None
    return stamp_root / filt / filename_template.format(filter=filt, id=obj_id)


def plot_gallery(
    rows: List[Dict],
    output_path: Path,
    title: str,
    samples_per_class: int,
    seed: int
) -> None:
    if not rows:
        logging.warning("No rows available for %s", title)
        return

    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for cls in FAMILY_LABELS:
            cls_rows = [row for row in rows if row['morph_class'] == cls]
            if not cls_rows:
                continue
            if len(cls_rows) > samples_per_class:
                idx = rng.choice(len(cls_rows), size=samples_per_class, replace=False)
                cls_rows = [cls_rows[i] for i in idx]

            n = len(cls_rows)
            ncols = min(5, n)
            nrows = int(math.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
            axes = np.atleast_1d(axes).ravel()

            for ax, row in zip(axes, cls_rows):
                try:
                    img = Image.open(row['file_loc']).convert('L')
                    ax.imshow(np.array(img), cmap='gray')
                    ax.set_title(f"{row['id']}\nz={row['z']:.2f} logM={row['mass']:.2f}")
                except Exception as exc:
                    ax.text(0.5, 0.5, f"Missing\n{exc}", ha='center', va='center')
                    ax.set_facecolor('lightgray')
                ax.axis('off')

            for ax in axes[n:]:
                ax.axis('off')

            fig.suptitle(f"{title} - {cls}", fontsize=14, y=1.02)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    logging.info("Saved gallery to %s", output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = read_master_catalog(args.catalog)
    catalog_ids = normalize_ids(catalog['id'])

    family_matrix = load_family_morphology(args.morphology, catalog_ids)
    for idx, col in enumerate(FAMILY_COLS):
        catalog[col] = family_matrix[:, idx]

    has_any_morph = np.isfinite(family_matrix).any(axis=1)
    catalog['has_any_morphology'] = has_any_morph

    final_mask, mass_log, z_vals, _ = build_clean_mask(catalog)

    scores_for_sort = np.nan_to_num(family_matrix, nan=-np.inf)
    main_idx = np.argmax(scores_for_sort, axis=1)
    morph_class = np.array(FAMILY_LABELS)[main_idx]
    catalog['morph_unique_class'] = morph_class

    group_mask, field_mask, _, _ = load_group_membership(args.group_dir, catalog_ids)

    base_mask = final_mask & catalog['has_any_morphology'] & np.isfinite(mass_log)

    mass_bins = []
    for part in args.mass_bins.split(','):
        lo_str, hi_str = part.split('-')
        mass_bins.append((float(lo_str), float(hi_str)))

    rng = np.random.default_rng(args.seed)
    matched = build_matched_masks(base_mask, mass_log, group_mask, field_mask, mass_bins, rng)

    stamp_root = args.stamp_root.expanduser().resolve()
    filename_template = "{filter}_{id}.jpg"

    for (lo, hi), (gmask, fmask) in matched.items():
        for env_name, env_mask in [('group', gmask), ('field', fmask)]:
            if env_mask.sum() == 0:
                logging.warning("No matched samples for %s %.1f-%.1f", env_name, lo, hi)
                continue

            rows = []
            env_indices = np.where(env_mask)[0]
            for idx in env_indices:
                obj_id = catalog_ids.iloc[idx]
                z = float(catalog['zfinal'].iloc[idx])
                mass = float(mass_log[idx])
                if not np.isfinite(z) or not np.isfinite(mass):
                    continue
                file_path = select_stamp_path(stamp_root, obj_id, z, filename_template)
                if file_path is None or not file_path.exists():
                    continue
                rows.append({
                    'id': obj_id,
                    'z': z,
                    'mass': mass,
                    'file_loc': str(file_path),
                    'morph_class': catalog['morph_unique_class'].iloc[idx]
                })

            title = f"{env_name} quenched {lo:.1f}-{hi:.1f}"
            output_path = output_dir / f"gallery_{env_name}_{lo:.1f}_{hi:.1f}.pdf"
            plot_gallery(rows, output_path, title, args.samples_per_class, args.seed)


if __name__ == "__main__":
    main()
