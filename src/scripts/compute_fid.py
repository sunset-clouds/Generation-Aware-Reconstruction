"""
Compute FID/IS from pre-generated NPZ files.

Supports two evaluator backends:
  openai:         OpenAI TF Inception V3 (VIRTUAL_imagenet256_labeled.npz)
  torch_fidelity: torch-fidelity PyTorch Inception V3 (jit_in256_stats.npz)

Auto-detects available NPZ files in --npz_dir and computes all available metrics.

Usage:
    # Both evaluators:
    python scripts/compute_fid.py \
        --npz_dir ./results/fid/baseline \
        --model_type iMF-B-2 \
        --evaluator both

    # OpenAI evaluator with custom reference:
    python scripts/compute_fid.py \
        --npz_dir ./results/fid/baseline \
        --model_type iMF-B-2 \
        --evaluator openai \
        --ref_npz metric/VIRTUAL_imagenet256_labeled.npz

    # torch-fidelity only:
    python scripts/compute_fid.py \
        --npz_dir ./results/fid/baseline \
        --model_type iMF-B-2 \
        --evaluator torch_fidelity
"""

import os
import sys
import json
import glob
import argparse
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OPENAI_REF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'metric', 'VIRTUAL_imagenet256_labeled.npz'
)
DEFAULT_TORCH_FIDELITY_REF = (
    os.environ.get(
        "JIT_IN256_STATS",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'metric', 'jit_in256_stats.npz'
        )
    )
)


def discover_npz_files(npz_dir, model_type):
    """
    Auto-discover which NPZ files are available and categorize them.
    Returns dict of {metric_name: npz_path}.
    """
    found = {}

    # Reference images
    input_npz = os.path.join(npz_dir, 'Input.npz')
    if os.path.exists(input_npz):
        found['input'] = input_npz

    # Vanilla rFID
    vae_npz = os.path.join(npz_dir, 'VAE_reconstruction.npz')
    if os.path.exists(vae_npz):
        found['vanilla_rfid'] = vae_npz

    # gFID with CFG
    cfg_npz = os.path.join(npz_dir, f'Generated_CFG_{model_type}.npz')
    if os.path.exists(cfg_npz):
        found['gfid_cfg'] = cfg_npz

    # gFID without CFG
    nocfg_npz = os.path.join(npz_dir, f'Generated_noCFG_{model_type}.npz')
    if os.path.exists(nocfg_npz):
        found['gfid_nocfg'] = nocfg_npz

    # Denoising rFID (pattern: Denoising_t{t}_{noCFG|CFG}_{model}.npz)
    denoise_pattern = os.path.join(npz_dir, f'Denoising_t*_*_{model_type}.npz')
    for path in sorted(glob.glob(denoise_pattern)):
        basename = os.path.basename(path)
        match = re.match(
            rf'Denoising_t([\d.]+)_(noCFG|CFG)_{re.escape(model_type)}\.npz',
            basename
        )
        if match:
            t_val = match.group(1)
            cfg_tag = match.group(2)
            key = f'our_rfid_{"nocfg" if cfg_tag == "noCFG" else "cfg"}_t{t_val}'
            found[key] = path

    return found


def compute_with_openai(npz_files, ref_npz):
    """Compute FID/IS using OpenAI TF Inception evaluator."""
    from metric.evaluator import FIDEvaluator

    evaluator = FIDEvaluator()
    results = {}

    try:
        for key, sample_npz in npz_files.items():
            if key == 'input':
                continue

            print(f"\n  [{key}] Computing FID (OpenAI TF Inception)...")
            print(f"    Reference: {ref_npz}")
            print(f"    Sample:    {sample_npz}")

            fid, is_score = evaluator.compute_fid_and_is(ref_npz, sample_npz)
            results[key] = {'fid': float(fid), 'is': float(is_score)}
            print(f"    FID: {fid:.4f}, IS: {is_score:.2f}")
    finally:
        evaluator.close()

    return results


def compute_with_torch_fidelity(npz_files, ref_stats):
    """Compute FID/IS using torch-fidelity PyTorch Inception evaluator."""
    from metric.fidelity import TorchFidelityEvaluator

    evaluator = TorchFidelityEvaluator()
    results = {}

    for key, sample_npz in npz_files.items():
        if key == 'input':
            continue

        print(f"\n  [{key}] Computing FID (torch-fidelity PyTorch Inception)...")
        print(f"    Reference: {ref_stats}")
        print(f"    Sample:    {sample_npz}")

        fid, is_mean = evaluator.compute_fid_and_is(sample_npz, ref_stats)
        results[key] = {'fid': float(fid), 'is': float(is_mean)}
        print(f"    FID: {fid:.4f}, IS: {is_mean:.2f}")

    return results


def print_summary_table(results, evaluator_name):
    """Print a formatted summary table."""
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — {evaluator_name}")
    print(f"{'=' * 70}")
    print(f"{'Metric':<35} {'FID':>10} {'IS':>10}")
    print("-" * 55)

    ordering = [
        'vanilla_rfid',
        'gfid_cfg', 'gfid_nocfg',
    ]

    printed = set()
    for key in ordering:
        if key in results:
            r = results[key]
            label = {
                'vanilla_rfid': 'Vanilla rFID',
                'gfid_cfg': 'gFID (with CFG)',
                'gfid_nocfg': 'gFID (no CFG)',
            }.get(key, key)
            print(f"  {label:<33} {r['fid']:>10.4f} {r['is']:>10.2f}")
            printed.add(key)

    # Denoising rFID entries
    for key in sorted(results.keys()):
        if key not in printed:
            r = results[key]
            label = key.replace('our_rfid_nocfg_', 'Our rFID (no CFG) ').replace(
                'our_rfid_cfg_', 'Our rFID (CFG) ')
            print(f"  {label:<33} {r['fid']:>10.4f} {r['is']:>10.2f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compute FID/IS from NPZ files")
    parser.add_argument('--npz_dir', type=str, required=True,
                        help='Directory containing NPZ files from generate_eval_images.py')
    parser.add_argument('--model_type', type=str, default='iMF-B-2',
                        choices=['iMF-B-2', 'iMF-M-2', 'iMF-L-2', 'iMF-XL-2'])
    parser.add_argument('--evaluator', type=str, default='both',
                        choices=['openai', 'torch_fidelity', 'both'],
                        help='Which FID evaluator backend to use')
    parser.add_argument('--ref_npz', type=str, default=None,
                        help='Override reference NPZ for OpenAI evaluator '
                             '(default: metric/VIRTUAL_imagenet256_labeled.npz)')
    parser.add_argument('--torch_fidelity_ref', type=str, default=None,
                        help='Override reference NPZ for torch-fidelity '
                             '(default: metric/jit_in256_stats.npz or JIT_IN256_STATS)')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Output JSON file (default: {npz_dir}/fid_results.json)')
    args = parser.parse_args()

    # Discover available NPZ files
    npz_files = discover_npz_files(args.npz_dir, args.model_type)

    print("=" * 70)
    print("FID/IS Computation")
    print("=" * 70)
    print(f"NPZ directory: {args.npz_dir}")
    print(f"Model:         {args.model_type}")
    print(f"Evaluator:     {args.evaluator}")
    print(f"\nDiscovered NPZ files:")
    for key, path in sorted(npz_files.items()):
        print(f"  {key:<35} {os.path.basename(path)}")
    print("=" * 70)

    sample_keys = [key for key in npz_files.keys() if key != 'input']
    if not sample_keys:
        print("No sample NPZ files found (only Input.npz or nothing). "
              "Run generate_eval_images.py first.")
        sys.exit(1)

    all_results = {
        'npz_dir': args.npz_dir,
        'model_type': args.model_type,
    }

    # OpenAI evaluator
    if args.evaluator in ('openai', 'both'):
        ref_npz = args.ref_npz or DEFAULT_OPENAI_REF

        # For gFID, use pre-computed reference stats (VIRTUAL_imagenet256_labeled.npz)
        # For rFID, use Input.npz as reference
        gfid_npz = {k: v for k, v in npz_files.items()
                    if k.startswith('gfid')}
        rfid_npz = {k: v for k, v in npz_files.items()
                    if k in ('vanilla_rfid',) or k.startswith('our_rfid')}

        openai_results = {}

        if rfid_npz:
            input_npz = npz_files.get('input')
            if input_npz:
                print("\n--- OpenAI Evaluator: rFID (ref = Input.npz) ---")
                openai_results.update(compute_with_openai(rfid_npz, input_npz))
            else:
                print("WARNING: Input.npz not found, skipping rFID computation")

        if gfid_npz:
            if not os.path.exists(ref_npz):
                print(f"WARNING: Reference NPZ not found at {ref_npz}")
                print("  For gFID, provide --ref_npz pointing to "
                      "VIRTUAL_imagenet256_labeled.npz")
            else:
                print(f"\n--- OpenAI Evaluator: gFID (ref = {os.path.basename(ref_npz)}) ---")
                openai_results.update(compute_with_openai(gfid_npz, ref_npz))

        if openai_results:
            print_summary_table(openai_results, "OpenAI TF Inception")
            all_results['openai'] = openai_results

    # torch-fidelity evaluator
    if args.evaluator in ('torch_fidelity', 'both'):
        tf_ref = args.torch_fidelity_ref or DEFAULT_TORCH_FIDELITY_REF

        # torch-fidelity only supports gFID (reference is jit_in256_stats.npz)
        # For rFID, it would need Input.npz converted — use OpenAI evaluator instead
        gfid_npz = {k: v for k, v in npz_files.items()
                    if k.startswith('gfid')}
        rfid_npz = {k: v for k, v in npz_files.items()
                    if k in ('vanilla_rfid',) or k.startswith('our_rfid')}

        tf_results = {}

        if gfid_npz:
            print(f"\n--- torch-fidelity Evaluator: gFID (ref = jit_in256_stats) ---")
            tf_results.update(compute_with_torch_fidelity(gfid_npz, tf_ref))

        if rfid_npz:
            input_npz = npz_files.get('input')
            if input_npz:
                print(f"\n--- torch-fidelity Evaluator: rFID (ref = Input.npz) ---")
                tf_results.update(compute_with_torch_fidelity(rfid_npz, input_npz))

        if tf_results:
            print_summary_table(tf_results, "torch-fidelity PyTorch Inception")
            all_results['torch_fidelity'] = tf_results

    # Save results
    output_json = args.output_json or os.path.join(args.npz_dir, 'fid_results.json')
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_json}")


if __name__ == '__main__':
    main()
