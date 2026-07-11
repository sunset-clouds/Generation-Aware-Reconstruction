from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.run_sit_garfid_single import (  # noqa: E402
    collect_names,
    resolve_user_path,
    save_npz,
    save_uint8_batch,
)
from scripts.imagenet_val_labeled_dataset import PairedImageNetDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the ImageNet reference PNG set used by SiT GAR-FID runs. "
            "This is useful for GAR-only sweeps that want to reuse one shared input_png directory."
        )
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--label_map_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_npz", action="store_true")
    return parser.parse_args()


def count_pngs(path: Path) -> int:
    return sum(1 for candidate in path.iterdir() if candidate.is_file() and candidate.suffix.lower() == ".png")


def resolve_path(path_value: str, *, extra_roots: Iterable[Path] | None = None) -> str:
    roots = [WORKSPACE_ROOT]
    if extra_roots:
        roots.extend(extra_roots)
    return resolve_user_path(path_value, extra_roots=roots)


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_path(args.dataset_dir)
    label_map_json = resolve_path(args.label_map_json)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PairedImageNetDataset(
        dataset_dir,
        num_samples=args.num_samples,
        label_map_json=label_map_json,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
    )

    npz_chunks: List[np.ndarray] = []
    processed = 0
    for batch in tqdm(dataloader, desc="Export reference PNGs"):
        _, batch_uint8, batch_indices, _ = batch
        indices_list = [int(i) for i in batch_indices.tolist()]
        names = collect_names(dataset, indices_list)
        images = batch_uint8.numpy()
        save_uint8_batch(output_dir, names, indices_list, images, "original")
        if args.save_npz:
            npz_chunks.append(images)
        processed += len(indices_list)

    if args.save_npz:
        images = (
            np.concatenate(npz_chunks, axis=0)
            if npz_chunks
            else np.empty((0, 256, 256, 3), dtype=np.uint8)
        )
        save_npz(output_dir.parent / "Input.npz", images)

    summary = {
        "dataset_dir": dataset_dir,
        "label_map_json": label_map_json,
        "output_dir": str(output_dir),
        "num_samples_requested": int(args.num_samples),
        "num_samples_exported": int(processed),
        "png_count": int(count_pngs(output_dir)),
    }
    summary_path = output_dir.parent / "reference_export_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 72)
    print("Reference PNG export finished")
    print(f"dataset_dir : {dataset_dir}")
    print(f"output_dir  : {output_dir}")
    print(f"exported    : {processed}")
    print(f"png_count   : {summary['png_count']}")
    print(f"summary     : {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
