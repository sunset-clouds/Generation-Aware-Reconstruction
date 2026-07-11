import json
import os
import re


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_CLASS_INDEX_JSON = os.path.join(SCRIPT_DIR, "imagenet_class_index.json")
DEFAULT_LABEL_MAP_JSON = os.path.join(SCRIPT_DIR, "imagenet_val_dir_to_index.json")

# These names are known collisions in human-readable ImageNet folder dumps.
# We pin them explicitly so the generated label map is exact and stable.
EXPLICIT_DIR_OVERRIDES = {
    "bathtub, bathing tub, bath, tub": 435,
    "cardigan": 474,
    "chime, bell, gong": 494,
    "crane": 134,
    "harmonica, mouth organ, harp, mouth harp": 593,
    "maillot": 638,
    "projectile, missile": 744,
    "ruffed grouse, partridge, Bonasa umbellus": 82,
    "skunk, polecat, wood pussy": 361,
    "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish": 123,
    "Cardigan, Cardigan Welsh corgi": 264,
    "crane2": 517,
    "maillot, tank suit": 639,
}


def canonicalize_label_name(name):
    """Normalize names like 'great white shark' and 'great_white_shark'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_official_class_index(mapping_path=OFFICIAL_CLASS_INDEX_JSON):
    """Load the bootstrap ImageNet-1k index used to generate exact folder maps."""
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Missing ImageNet class index file: {mapping_path}")

    with open(mapping_path, "r") as f:
        raw_mapping = json.load(f)

    synset_to_idx = {}
    human_to_idx = {}
    ambiguous_human_labels = set()

    for idx_str, pair in raw_mapping.items():
        idx = int(idx_str)
        synset, human_label = pair
        synset_to_idx[synset] = idx

        canonical_human = canonicalize_label_name(human_label)
        if canonical_human in human_to_idx:
            ambiguous_human_labels.add(canonical_human)
        else:
            human_to_idx[canonical_human] = idx

    for canonical_human in ambiguous_human_labels:
        human_to_idx.pop(canonical_human, None)

    return synset_to_idx, human_to_idx, ambiguous_human_labels


def candidate_indices_from_directory_name(class_dir, human_to_idx):
    """
    Recover an ImageNet class index from a human-readable directory name.

    Supports both simplified labels like "accordion" and original synset-words
    strings like "accordion, piano accordion, squeeze box".
    """
    candidates = set()

    canonical_full = canonicalize_label_name(class_dir)
    if canonical_full in human_to_idx:
        candidates.add(human_to_idx[canonical_full])

    if "," in class_dir:
        for alias in class_dir.split(","):
            canonical_alias = canonicalize_label_name(alias)
            if canonical_alias in human_to_idx:
                candidates.add(human_to_idx[canonical_alias])

    return candidates


def resolve_directory_name(class_dir, synset_to_idx, human_to_idx, ambiguous_human_labels):
    """Resolve one dataset directory name to an ImageNet-1k class index."""
    if class_dir.isdigit():
        return int(class_dir), "numeric"

    if class_dir in EXPLICIT_DIR_OVERRIDES:
        return EXPLICIT_DIR_OVERRIDES[class_dir], "override"

    if class_dir in synset_to_idx:
        return synset_to_idx[class_dir], "synset"

    canonical_class_dir = canonicalize_label_name(class_dir)
    if canonical_class_dir in ambiguous_human_labels:
        return None, "ambiguous"

    dir_candidates = candidate_indices_from_directory_name(class_dir, human_to_idx)
    if len(dir_candidates) == 1:
        return next(iter(dir_candidates)), "human_label"
    if len(dir_candidates) > 1:
        return None, "ambiguous"
    return None, "unresolved"


def list_class_directories(dataset_dir):
    """Return sorted class subdirectories under an ImageNet-style root."""
    class_dirs = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    if not class_dirs:
        raise ValueError(f"No class subdirectories found under {dataset_dir}")
    return sorted(class_dirs)


def build_exact_label_map(dataset_dir):
    """Infer an exact directory-name -> class-index map for one dataset root."""
    class_dirs = list_class_directories(dataset_dir)
    synset_to_idx, human_to_idx, ambiguous_human_labels = load_official_class_index()

    resolved = []
    ambiguous = []
    unresolved = []
    label_map = {}

    for class_dir in class_dirs:
        resolved_idx, reason = resolve_directory_name(
            class_dir, synset_to_idx, human_to_idx, ambiguous_human_labels
        )
        if resolved_idx is None:
            if reason == "ambiguous":
                ambiguous.append(class_dir)
            else:
                unresolved.append(class_dir)
            continue

        label_map[class_dir] = int(resolved_idx)
        resolved.append({
            "class_dir": class_dir,
            "index": int(resolved_idx),
            "reason": reason,
        })

    duplicate_target_indices = _duplicate_target_indices(label_map)
    report = {
        "dataset_dir": dataset_dir,
        "num_class_directories": len(class_dirs),
        "resolved": resolved,
        "ambiguous": ambiguous,
        "unresolved": unresolved,
        "duplicate_target_indices": duplicate_target_indices,
    }
    return label_map, report


def load_exact_label_map(label_map_json=DEFAULT_LABEL_MAP_JSON):
    """Load a hand-checked exact directory-name -> class-index mapping."""
    if not os.path.exists(label_map_json):
        raise FileNotFoundError(
            f"Missing exact label map: {label_map_json}. "
            "Generate it first with scripts/build_imagenet_label_map.py."
        )

    with open(label_map_json, "r") as f:
        raw_map = json.load(f)

    label_map = {str(k): int(v) for k, v in raw_map.items()}
    invalid = [name for name, idx in label_map.items() if idx < 0 or idx >= 1000]
    if invalid:
        raise ValueError(
            f"Exact label map contains invalid indices for: {invalid[:5]}"
        )
    return label_map


def audit_exact_label_map(dataset_dir, label_map):
    """Audit an exact map against a dataset root."""
    class_dirs = list_class_directories(dataset_dir)
    dataset_class_dirs = set(class_dirs)
    map_class_dirs = set(label_map.keys())

    missing_from_map = sorted(dataset_class_dirs - map_class_dirs)
    extra_in_map = sorted(map_class_dirs - dataset_class_dirs)
    duplicate_target_indices = _duplicate_target_indices(
        {name: label_map[name] for name in class_dirs if name in label_map}
    )
    invalid_indices = sorted(
        name for name, idx in label_map.items() if idx < 0 or idx >= 1000
    )

    resolved = [
        {"class_dir": name, "index": int(label_map[name]), "reason": "exact_map"}
        for name in class_dirs
        if name in label_map
    ]
    return {
        "dataset_dir": dataset_dir,
        "num_class_directories": len(class_dirs),
        "resolved": resolved,
        "missing_from_map": missing_from_map,
        "extra_in_map": extra_in_map,
        "duplicate_target_indices": duplicate_target_indices,
        "invalid_indices": invalid_indices,
    }


def print_generated_map_report(report, preview_count=20):
    """Pretty-print build-time resolution results."""
    print("\n--- Label Map Build Report ---")
    print(f"dataset_dir: {report['dataset_dir']}")
    print(f"class directories: {report['num_class_directories']}")
    print(f"resolved: {len(report['resolved'])}")
    print(f"ambiguous: {len(report['ambiguous'])}")
    print(f"unresolved: {len(report['unresolved'])}")
    print(f"duplicate target indices: {len(report['duplicate_target_indices'])}")

    if report["ambiguous"]:
        print("\nAmbiguous class directories:")
        for name in report["ambiguous"][:preview_count]:
            print(f"  - {name}")

    if report["unresolved"]:
        print("\nUnresolved class directories:")
        for name in report["unresolved"][:preview_count]:
            print(f"  - {name}")

    if report["duplicate_target_indices"]:
        print("\nDuplicate target indices:")
        for idx_str, dirs in list(report["duplicate_target_indices"].items())[:preview_count]:
            print(f"  - {idx_str}: {dirs}")

    print("\nResolved mapping preview:")
    for item in report["resolved"][:preview_count]:
        print(f"  - {item['class_dir']} -> {item['index']} ({item['reason']})")

    ok = (
        not report["ambiguous"]
        and not report["unresolved"]
        and not report["duplicate_target_indices"]
    )
    print(f"\nBuild check passed: {ok}")
    return ok


def print_exact_map_audit(report, preview_count=20):
    """Pretty-print exact-map audit results."""
    print("\n--- Exact Label Map Audit ---")
    print(f"dataset_dir: {report['dataset_dir']}")
    print(f"class directories: {report['num_class_directories']}")
    print(f"resolved from map: {len(report['resolved'])}")
    print(f"missing from map: {len(report['missing_from_map'])}")
    print(f"extra in map: {len(report['extra_in_map'])}")
    print(f"duplicate target indices: {len(report['duplicate_target_indices'])}")
    print(f"invalid indices: {len(report['invalid_indices'])}")

    if report["missing_from_map"]:
        print("\nMissing from map:")
        for name in report["missing_from_map"][:preview_count]:
            print(f"  - {name}")

    if report["extra_in_map"]:
        print("\nExtra in map:")
        for name in report["extra_in_map"][:preview_count]:
            print(f"  - {name}")

    if report["duplicate_target_indices"]:
        print("\nDuplicate target indices:")
        for idx_str, dirs in list(report["duplicate_target_indices"].items())[:preview_count]:
            print(f"  - {idx_str}: {dirs}")

    if report["invalid_indices"]:
        print("\nInvalid indices:")
        for name in report["invalid_indices"][:preview_count]:
            print(f"  - {name}")

    print("\nResolved mapping preview:")
    for item in report["resolved"][:preview_count]:
        print(f"  - {item['class_dir']} -> {item['index']}")

    ok = (
        not report["missing_from_map"]
        and not report["extra_in_map"]
        and not report["duplicate_target_indices"]
        and not report["invalid_indices"]
    )
    print(f"\nExact-map check passed: {ok}")
    return ok


def save_label_map(label_map, output_json):
    """Write the exact map as a clean, reviewable JSON file."""
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    ordered = {name: int(label_map[name]) for name in sorted(label_map)}
    with open(output_json, "w") as f:
        json.dump(ordered, f, indent=2, sort_keys=False)


def _duplicate_target_indices(label_map):
    index_to_dirs = {}
    for class_dir, idx in label_map.items():
        index_to_dirs.setdefault(int(idx), []).append(class_dir)
    return {
        str(idx): dirs
        for idx, dirs in sorted(index_to_dirs.items())
        if len(dirs) > 1
    }


def audit_legacy_analyze_latents_synthetic_labels(dataset_dir, num_samples=50000, num_classes=1000):
    """
    Check the **original** ``analyze_latents_and_features`` label recipe against folder semantics.

    **Structure under test (legacy):**

    1. Image order: same as ``ImageNetDataset`` / ``list_imagenet_val_image_paths``.
    2. Labels passed to ``generate``: ``all_labels[idx]`` with
       ``all_labels = torch.arange(1000).repeat_interleave(num_samples // 1000)[:num_samples]``.

    **Reference class index** for each path: parent folder name resolved by
    ``build_exact_label_map`` (``imagenet_class_index.json`` + ``EXPLICIT_DIR_OVERRIDES``).
    This does **not** load ``imagenet_val_dir_to_index.json``; that JSON should match this
    resolution if generated from the same val tree.

    A high mismatch rate means: synthetic labels do not describe the class of the image at
    the same index (standard val is folder-sorted, not "50×class0, 50×class1, …" in index order).
    """
    import torch

    from scripts.imagenet_val_paths import list_imagenet_val_image_paths

    truth_map, build_report = build_exact_label_map(dataset_dir)
    paths = list_imagenet_val_image_paths(dataset_dir, num_samples)
    n_paths = len(paths)
    rep = num_samples // num_classes
    all_labels = torch.arange(num_classes).repeat_interleave(rep)[:num_samples]
    n_lab = int(all_labels.numel())
    compared = min(n_paths, n_lab)

    mismatches = []
    missing_in_truth = []
    for idx in range(compared):
        parent = os.path.basename(os.path.dirname(paths[idx]))
        if parent not in truth_map:
            missing_in_truth.append((idx, parent))
            continue
        true_y = truth_map[parent]
        syn_y = int(all_labels[idx].item())
        if true_y != syn_y:
            mismatches.append((idx, parent, true_y, syn_y))

    dup = build_report.get("duplicate_target_indices") or {}
    return {
        "dataset_dir": dataset_dir,
        "num_samples_arg": num_samples,
        "repeat_interleave_width": rep,
        "n_paths": n_paths,
        "n_synthetic_labels": n_lab,
        "n_compared": compared,
        "n_mismatch": len(mismatches),
        "n_missing_truth_for_parent": len(missing_in_truth),
        "mismatch_rate_among_compared": (len(mismatches) / compared) if compared else 0.0,
        "first_mismatches": mismatches[:25],
        "first_missing_parents": missing_in_truth[:10],
        "build_report_ambiguous": len(build_report.get("ambiguous") or []),
        "build_report_unresolved": len(build_report.get("unresolved") or []),
        "build_report_duplicate_indices": len(dup),
    }


def print_audit_legacy_analyze_latents_synthetic_labels(result):
    print("=== Legacy analyze_latents synthetic label schedule (structure-only audit) ===")
    print(f"dataset_dir: {result['dataset_dir']}")
    print(f"num_samples (arg): {result['num_samples_arg']}")
    print(f"repeat_interleave width: {result['repeat_interleave_width']} (num_samples // 1000)")
    print(f"paths enumerated: {result['n_paths']}, synthetic tensor length: {result['n_synthetic_labels']}")
    print(f"compared indices: {result['n_compared']}")
    print(
        f"build_exact_label_map: ambiguous={result['build_report_ambiguous']}, "
        f"unresolved={result['build_report_unresolved']}, "
        f"duplicate_indices={result['build_report_duplicate_indices']}"
    )
    print(
        f"mismatches (folder-implied class != synthetic all_labels[idx]): "
        f"{result['n_mismatch']} ({100.0 * result['mismatch_rate_among_compared']:.2f}%)"
    )
    print(f"missing parent in build_exact_label_map: {result['n_missing_truth_for_parent']}")
    if result["first_mismatches"]:
        print("\nFirst mismatches (idx, parent_dir, class_from_folder, synthetic_label):")
        for row in result["first_mismatches"][:15]:
            print(f"  {row}")
    if result["first_missing_parents"]:
        print("\nFirst paths whose parent folder was not in build_exact_label_map:")
        for row in result["first_missing_parents"]:
            print(f"  {row}")
    print(
        "\nInterpretation: JSON file is not the authority here — "
        "this compares legacy tensor indexing to **folder-name class index** "
        "from the same rules as build_imagenet_label_map.py."
    )
