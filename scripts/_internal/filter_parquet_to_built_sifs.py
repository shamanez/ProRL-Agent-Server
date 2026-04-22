#!/usr/bin/env python3
"""Filter a SkyRL-style parquet down to the rows whose SIF already exists.

When the trainer launches against a partial SIF build (e.g. only 30/293
images are ready), it would otherwise throw `Singularity image not found`
on every missing one. This helper writes a new parquet containing only the
rows whose expected .sif file is present in the image cache.

Naming mapping mirrors `scripts/pull_swe_images.py::instance_id_to_image`:
    instance_id  getmoto__moto-4950
    image        xingyaoww/sweb.eval.x86_64.getmoto_s_moto-4950
    sanitized    xingyaoww_sweb.eval.x86_64.getmoto_s_moto-4950
    sif path     <sif_dir>/xingyaoww_sweb.eval.x86_64.getmoto_s_moto-4950.sif

Usage:
    /opt/pytorch/bin/python3 scripts/_internal/filter_parquet_to_built_sifs.py \
        --source ~/data/SkyRL-v0-293/train.parquet \
        --sif-dir /home/ubuntu/.../singularity_images \
        --dest ~/data/SkyRL-v0-293/train.filtered.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_PREFIX = 'xingyaoww'


def instance_id_to_sif_name(instance_id: str, prefix: str = DEFAULT_PREFIX) -> str:
    """Replicate pull_swe_images.instance_id_to_image + sanitize, non-multimodal."""
    suffix = instance_id.replace('__', '_s_').lower()
    core = f'sweb.eval.x86_64.{suffix}'
    image = f'{prefix.rstrip("/")}/{core}'.lower()
    return f'{image.replace("/", "_").replace(":", "_")}.sif'


def _extract_instance_id(row_instance) -> str | None:
    if isinstance(row_instance, dict):
        return row_instance.get('instance_id')
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Input parquet')
    p.add_argument('--dest', type=Path, required=True, help='Output parquet')
    p.add_argument(
        '--sif-dir', type=Path, required=True, help='Singularity image cache'
    )
    p.add_argument(
        '--prefix',
        default=DEFAULT_PREFIX,
        help='Docker image prefix (default xingyaoww)',
    )
    p.add_argument(
        '--min-rows', type=int, default=1, help='Fail if fewer rows survive the filter'
    )
    args = p.parse_args()

    if not args.source.exists():
        print(f'[ERROR] Source parquet not found: {args.source}', file=sys.stderr)
        return 2
    if not args.sif_dir.is_dir():
        print(f'[ERROR] SIF dir not found: {args.sif_dir}', file=sys.stderr)
        return 2

    df = pd.read_parquet(args.source)
    if 'instance' not in df.columns:
        print(
            f"[ERROR] Expected 'instance' column; got {list(df.columns)}",
            file=sys.stderr,
        )
        return 2

    existing = {p.name for p in args.sif_dir.glob('*.sif')}
    print(f'[INFO] {len(existing)} SIFs in {args.sif_dir}')

    keep_mask = []
    missing_examples: list[str] = []
    for inst in df['instance']:
        iid = _extract_instance_id(inst)
        if iid is None:
            keep_mask.append(False)
            continue
        sif_name = instance_id_to_sif_name(iid, prefix=args.prefix)
        present = sif_name in existing
        keep_mask.append(present)
        if not present and len(missing_examples) < 3:
            missing_examples.append(sif_name)

    filtered = df[keep_mask].reset_index(drop=True)
    print(f'[INFO] {len(filtered)}/{len(df)} rows survive filter')
    if missing_examples:
        print(f'[INFO] Sample missing SIFs: {missing_examples}')

    if len(filtered) < args.min_rows:
        print(
            f'[ERROR] Only {len(filtered)} rows survive — need >= {args.min_rows}.'
            ' Build more SIFs before launching.',
            file=sys.stderr,
        )
        return 1

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(args.dest, index=False)
    print(f'[OK] Wrote {args.dest} ({len(filtered)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
