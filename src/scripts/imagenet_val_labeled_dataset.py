"""ImageNet val with per-image class index from exact folder-name map (shared by analysis scripts)."""
import os
import sys

from PIL import Image
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.imagenet_val_crop import center_crop_arr
from scripts.imagenet_dir_label_map import (
    DEFAULT_LABEL_MAP_JSON,
    audit_exact_label_map,
    load_exact_label_map,
)


class PairedImageNetDataset(torch.utils.data.Dataset):
    """Same traversal as ImageNetDataset; labels from label_map_json (audit on init)."""

    def __init__(self, root_dir, num_samples=50000, image_size=256, label_map_json=DEFAULT_LABEL_MAP_JSON):
        self.root_dir = root_dir
        self.image_size = image_size
        self.samples = []
        image_extensions = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}

        self.label_map = load_exact_label_map(label_map_json)
        audit = audit_exact_label_map(root_dir, self.label_map)
        ok = (
            not audit["missing_from_map"]
            and not audit["extra_in_map"]
            and not audit["duplicate_target_indices"]
            and not audit["invalid_indices"]
        )
        if not ok:
            raise ValueError(
                "Exact label map audit failed. Regenerate the map with "
                "scripts/build_imagenet_label_map.py before running analysis."
            )

        class_dirs = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.label_mapping_description = (
            f"exact directory-to-index map ({os.path.basename(label_map_json)})"
        )

        for class_dir in class_dirs:
            subdir_path = os.path.join(root_dir, class_dir)
            class_idx = self.label_map[class_dir]
            for fname in sorted(os.listdir(subdir_path)):
                if os.path.splitext(fname)[1] in image_extensions:
                    self.samples.append((os.path.join(subdir_path, fname), class_idx))

        self.samples = self.samples[:num_samples]
        if not self.samples:
            raise RuntimeError(f"No images found under {root_dir}")

        self._label_min = min(lab for _, lab in self.samples)
        self._label_max = max(lab for _, lab in self.samples)
        self._unique_labels = len({lab for _, lab in self.samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        img = Image.open(image_path).convert("RGB")
        arr = center_crop_arr(img, self.image_size)
        arr_uint8 = arr.copy()
        arr_float = arr.astype("float32") / 127.5 - 1.0
        img_tensor = torch.from_numpy(arr_float).permute(2, 0, 1)
        return img_tensor, arr_uint8, idx, label
