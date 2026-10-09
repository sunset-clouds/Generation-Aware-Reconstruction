# Entrypoint naming

The public interface follows the paper's two uses of generation-aware reconstruction:
**GAR-FID evaluation** and **decoder adaptation**. These are independent workflows,
not consecutive training stages.

## Commands

Run the commands from the repository root. The old forwarding entrypoints have
been removed; use the replacements below. The canonical Slurm templates retain
their scheduler directives.

| Previous entrypoint | Current entrypoint |
| --- | --- |
| `scripts/stage2_download_ifid_assets.sh` | `scripts/download_sit_checkpoints.sh` |
| `scripts/stage2_run_sit_garfid.sh` | `scripts/evaluate_garfid.sh` |
| `scripts/stage2_run_sit_garfid_sweep.sh` | `scripts/evaluate_garfid_sweep.sh` |
| `scripts/stage2_compute_correlations.sh` | `scripts/compute_correlations.sh` |
| `scripts/stage2_smoke_test.sh` | `scripts/smoke_test_garfid.sh` |
| `scripts/stage3_train_decoder_adaptation.sh` | `scripts/train_decoder_adaptation.sh` |
| `scripts/stage3_evaluate_posttrained_decoder.sh` | `scripts/evaluate_decoder.sh` |
| `scripts/stage3_download_hf_checkpoint.sh` | `scripts/download_hf_checkpoint.sh` |
| `scripts/generated/sbatch_stage2_smoke_server.sh` | `scripts/generated/sbatch_garfid_smoke_server.sh` |
| `scripts/generated/sbatch_stage3_smoke_server.sh` | `scripts/generated/sbatch_decoder_adaptation_smoke_server.sh` |
| `src/post_train.py` | `src/train_decoder_adaptation.py` |

Python imports should use `pipelines.sit_common`; the previous
`pipelines.sit_stage2_common` compatibility module has been removed.

## Evaluation modes and results

Use `garfid_nocfg` and `garfid_cfg` instead of `our_rfid_nocfg` and `our_rfid_cfg`.
Both reconstruction evaluation scripts accept the old mode names as aliases.
New result JSON keys and reconstruction-metric CSV mode values use the canonical
`garfid_*` names. Consumers of these fields should update their lookups.

Sample NPZ names (`Denoising_t*.npz`, `VAE_reconstruction.npz`, and
`Generated_*.npz`) are unchanged, so existing samples can be evaluated again.
Official-backend checkpoint state keys, loading, and resume behavior are unchanged.
New default training filenames use the prefix `GAR_Decoder_Adaptation_`;
use `--saver_name_pre` to retain a custom filename prefix.

## iMF checkpoints

iMF training and evaluation now require official `.pth` checkpoints. The legacy
converted `.pt` backend, JAX conversion tools, and both checkpoint-conversion
shell entrypoints have been removed. Download an official `.pth` checkpoint;
renaming a converted `.pt` file does not convert its architecture or weights.
Adapted decoder checkpoints still use `.pth.tar` and the same state keys.
Existing tokenizer/decoder weights remain loadable through `--pretrained_decoder`,
including checkpoints trained with the removed backend. Full training resume
requires a checkpoint saved with the official backend.
SiT GAR-FID checkpoints keep their original format.

## SiT assets

The default registry now points to `assets/sit/`. When a requested path there is
absent, the loader and asset utilities reuse the corresponding existing path under
`assets/stage2_assets/`. No downloaded files are moved.

Use `SIT_ASSET_REPO_ID`, `SIT_ASSETS_ROOT`, and `--sit_assets_root` in new
commands. The previous `STAGE2_ASSET_REPO_ID`, `STAGE2_ASSETS_ROOT`, and
`--stage2_assets_root` names remain accepted; the new environment variables take
precedence. As before, explicit `exp_path` entries in a registry take precedence
over the download-root default. For custom storage, supply a matching registry.

Third-party tokenizer implementations keep their own stage terminology.
