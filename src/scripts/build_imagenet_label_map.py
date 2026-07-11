import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.imagenet_dir_label_map import (
    DEFAULT_LABEL_MAP_JSON,
    build_exact_label_map,
    print_generated_map_report,
    save_label_map,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build a clean exact directory-name -> ImageNet class-index map."
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument(
        "--output_json",
        type=str,
        default=DEFAULT_LABEL_MAP_JSON,
        help="Where to save the clean exact map JSON.",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default=None,
        help="Optional path to save the build report JSON.",
    )
    parser.add_argument(
        "--preview_count",
        type=int,
        default=20,
        help="How many examples to print from the generated report.",
    )
    args = parser.parse_args()

    label_map, report = build_exact_label_map(args.dataset_dir)
    ok = print_generated_map_report(report, preview_count=max(1, args.preview_count))

    if args.report_json:
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved build report to {args.report_json}")

    if not ok:
        raise SystemExit(1)

    save_label_map(label_map, args.output_json)
    print(f"Saved clean exact label map to {args.output_json}")


if __name__ == "__main__":
    main()
