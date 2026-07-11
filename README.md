# GAR-FID: Aligning Reconstruction with Generation

This repository contains the Stage 2 and Stage 3 code for **GAR-FID**
(*Generation-Aligned Reconstruction FID*), used in the project
**Aligning Reconstruction with Generation**.

The released code focuses on:

- **Stage 2:** GAR-FID evaluation for SiT/iFID checkpoints.
- **Stage 3:** decoder adaptation / post-training for iMF models.

Large files are intentionally not committed. ImageNet, generated samples,
FID statistics, VAE checkpoints, SiT checkpoints, iMF checkpoints, and
post-trained checkpoints must be supplied through local paths or downloaded
with the commands below.

## Repository Layout

```text
assets/
  sit_checkpoint_registry_template.csv   Stage-2 SiT/iFID checkpoint registry
configs/stage3_decoder_adaptation/       Reference iMF-B/M/L/XL settings
scripts/
  prepare_imagenet_labels.sh             Build ImageNet val folder -> class id map
  stage2_*.sh                            Stage-2 GAR-FID wrappers
  stage3_*.sh                            Stage-3 conversion, training, eval wrappers
src/
  integrations/sit/                      SiT checkpoint loading and GAR adapters
  pipelines/                             Stage-2 GAR-FID pipelines
  post_train.py                          Stage-3 decoder-adaptation training
  data, models, metric, utils/           Shared runtime modules
third_party/ifid/                        Minimal iFID/SiT runtime used by Stage 2
requirements.txt                         Environment snapshot used in our runs
environment_full_stage3.txt              Stage-3 environment snapshot
```

## Quick Start Checklist

Before running experiments, prepare the following:

```text
1. Python environment with CUDA PyTorch
2. ImageNet train and validation folders
3. ImageNet validation label map JSON
4. ImageNet FID reference statistics NPZ
5. PyTorch FID Inception checkpoint
6. Stage-2 SiT/iFID args.json and checkpoints
7. Stage-3 iMF PyTorch checkpoint
8. Stage-3 SD-VAE, LPIPS, and DINO checkpoints
```

A typical local asset layout is:

```text
assets/
  generated/
    imagenet_val_dir_to_index.json
  stage2_assets/
    sit-b-sdvae-400k/
      args.json
      checkpoints/0400000.pt
  checkpoints/
    imf-gar/
      official-pytorch/iMF-B-2.pt
      official-pytorch/iMF-M-2.pt
      official-pytorch/iMF-L-2.pt
      official-pytorch/iMF-XL-2.pt
    pt_inception-2015-12-05-6726825d.pth
    dino_deitsmall16_pretrain.pth
    vgg.pth
  fid_stats/
    VIRTUAL_imagenet256_labeled.npz
```

Your ImageNet directory should contain either:

```text
/path/to/imagenet/
  train/
  val/
```

or, for our server preprocessed layout:

```text
/mnt/nfs/wenjie/dataset/imagenet_idx/
  train/
  val/
```

For a fresh clone, set these paths once and reuse them in the commands below:

```bash
export IMAGENET_ROOT=/path/to/imagenet
export IMAGENET_VAL=${IMAGENET_ROOT}/val
export LABEL_MAP_JSON=assets/generated/imagenet_val_dir_to_index.json
export FID_REFERENCE_FILE=assets/fid_stats/VIRTUAL_imagenet256_labeled.npz
export PYTORCH_FID_INCEPTION_CKPT=assets/checkpoints/pt_inception-2015-12-05-6726825d.pth
export LPIPS_VGG_CKPT=assets/checkpoints/vgg.pth
export DINO_CKPT_PATH=assets/checkpoints/dino_deitsmall16_pretrain.pth
export SD_VAE_PATH=/path/to/sd-vae-ft-mse
```

Minimum assets by experiment:

| Experiment | Required assets |
| --- | --- |
| Stage 2 smoke/full | ImageNet `val/`, label map JSON, ImageNet FID reference NPZ, PyTorch FID Inception checkpoint, SiT/iFID checkpoint row |
| Stage 3 training | ImageNet `train/` and `val/`, official iMF PyTorch checkpoint, SD-VAE, LPIPS VGG checkpoint, DINO checkpoint |
| Stage 3 evaluation | ImageNet `val/`, official iMF PyTorch checkpoint, decoder-adapted checkpoint, label map JSON, ImageNet FID reference NPZ, OpenAI Inception graph |

ImageNet and ImageNet FID statistics are not redistributed by this repository.
Use your licensed ImageNet copy and the same FID statistics file as the
reported experiment when reproducing paper numbers.

## Installation

We recommend Python 3.10 or 3.11 with a CUDA-compatible PyTorch build.

Using `uv`:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -r requirements.txt
uv pip install -r third_party/ifid/requirements.txt
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r third_party/ifid/requirements.txt
```

If PyTorch from `requirements.txt` does not match your CUDA driver, install the
correct PyTorch wheel first, then install the remaining packages.

If you run Python entry points directly instead of the provided shell wrappers,
set:

```bash
export PYTHONPATH=src:third_party/ifid:${PYTHONPATH:-}
```

## Download Checkpoints

### Stage 3 iMF checkpoints

The official iMF-B/M/L/XL PyTorch checkpoints and GAR decoder-adapted
checkpoints are hosted here:

```text
https://huggingface.co/wenjieshu/iMF-GAR-checkpoints
```

Download the official iMF PyTorch checkpoints:

```bash
mkdir -p assets/checkpoints/imf-gar
huggingface-cli download wenjieshu/iMF-GAR-checkpoints \
  official-pytorch/iMF-B-2.pt \
  official-pytorch/iMF-M-2.pt \
  official-pytorch/iMF-L-2.pt \
  official-pytorch/iMF-XL-2.pt \
  --repo-type model \
  --local-dir assets/checkpoints/imf-gar
```

Optional: download the GAR decoder-adapted checkpoints:

```bash
huggingface-cli download wenjieshu/iMF-GAR-checkpoints \
  gar-decoder-noise025045/iMF-B-2_GAR_decoder_noise0.25-0.45_final.pth.tar \
  gar-decoder-noise025045/iMF-M-2_GAR_decoder_noise0.25-0.45_final.pth.tar \
  gar-decoder-noise025045/iMF-L-2_GAR_decoder_noise0.25-0.45_final.pth.tar \
  gar-decoder-noise025045/iMF-XL-2_GAR_decoder_noise0.25-0.45_final.pth.tar \
  --repo-type model \
  --local-dir assets/checkpoints/imf-gar
```

If the Hugging Face repo is private in your environment, log in first:

```bash
huggingface-cli login
```

Use a read-only token for collaborators.

### Stage 2 SiT/iFID checkpoints

Stage 2 uses the public iFID checkpoint repository:

```text
xutongda/Making-rFID-Predictive-of-Diffusion-gFID
```

Download the SiT args/checkpoint files listed in the registry:

```bash
bash scripts/stage2_download_ifid_assets.sh --only_should_run
```

To download only one row, for example `sdvae_b`:

```bash
bash scripts/stage2_download_ifid_assets.sh --group_ids sdvae_b
```

This writes files under the registry paths, usually:

```text
assets/stage2_assets/
```

If you need a mirror, custom cache, or private token:

```bash
export STAGE2_ASSET_REPO_ID=xutongda/Making-rFID-Predictive-of-Diffusion-gFID
export STAGE2_ASSETS_ROOT=assets/stage2_assets
export HF_TOKEN=your_read_token_for_private_repos
```

Leave `HF_TOKEN` unset for public downloads.

To inspect what would be downloaded without copying large files:

```bash
bash scripts/stage2_download_ifid_assets.sh --only_should_run --dry_run
```

### FID and loss checkpoints

The code expects local FID/loss assets. The exact filenames used in our
commands are:

```bash
export PYTORCH_FID_INCEPTION_CKPT=/path/to/pt_inception-2015-12-05-6726825d.pth
export LPIPS_VGG_CKPT=/path/to/vgg.pth
export DINO_CKPT_PATH=/path/to/dino_deitsmall16_pretrain.pth
```

For OpenAI-style TensorFlow FID used by Stage-3 evaluation, either place:

```text
src/metric/classify_image_graph_def.pb
```

or set:

```bash
export OPENAI_INCEPTION_GRAPH=/path/to/classify_image_graph_def.pb
```

The OpenAI FID backend requires TensorFlow. If your PyTorch training
environment does not include TensorFlow, keep using it for image generation and
set `PYTHON` to a TensorFlow-enabled Python for the final FID step:

```bash
export PYTHON=/path/to/tensorflow_env/bin/python
```

We do not redistribute ImageNet or ImageNet FID statistics. Provide your own
`VIRTUAL_imagenet256_labeled.npz` or reuse the project-server path shown below.

## Prepare ImageNet Labels

Stage 2 needs a validation-folder to class-id map. Build it once from your
ImageNet validation folder:

```bash
bash scripts/prepare_imagenet_labels.sh \
  "${IMAGENET_VAL}" \
  "${LABEL_MAP_JSON}"
```

Then either copy it to the default Stage-2 path:

```bash
cp assets/generated/imagenet_val_dir_to_index.json \
  src/scripts/imagenet_val_dir_to_index.json
```

or pass it through `LABEL_MAP_JSON` when running Stage 2. The commands below
use the environment variable and do not require copying into `src/scripts/`.

## Stage 2: GAR-FID on SiT/iFID Models

### Smoke test

Run a small Stage-2 check first:

```bash
ETA_T=0.3 NUM_SAMPLES=8 BATCH_SIZE=2 SAVE_NPZ=1 \
bash scripts/stage2_run_sit_garfid.sh \
  sdvae_b \
  "${IMAGENET_VAL}" \
  "${FID_REFERENCE_FILE}" \
  results/stage2_smoke/sdvae_b_eta03_n8
```

Expected output:

```text
results/stage2_smoke/sdvae_b_eta03_n8/summary.json
results/stage2_smoke/sdvae_b_eta03_n8/preview_grid.png
```

### Full single-point run

```bash
ETA_T=0.3 NUM_SAMPLES=50000 BATCH_SIZE=16 SAVE_NPZ=1 \
bash scripts/stage2_run_sit_garfid.sh \
  sdvae_b \
  "${IMAGENET_VAL}" \
  "${FID_REFERENCE_FILE}" \
  results/stage2_sit_garfid/sdvae_b_eta03
```

### Noise sweep

```bash
ETA_LIST="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0" \
NUM_SAMPLES=50000 BATCH_SIZE=16 SAVE_NPZ=1 \
bash scripts/stage2_run_sit_garfid_sweep.sh \
  sdvae_b \
  "${IMAGENET_VAL}" \
  "${FID_REFERENCE_FILE}" \
  results/stage2_sit_garfid/sdvae_b
```

### Correlation table

After collecting Stage-2 rows:

```bash
bash scripts/stage2_compute_correlations.sh \
  /path/to/stage2_garfid_results.csv \
  results/stage2_correlation_results.csv \
  results/stage2_correlation_rows.tex
```

## Stage 3: Decoder Adaptation for iMF

Stage 3 trains the SD-VAE decoder with the iMF latent generator fixed.

Important: `scripts/stage3_train_decoder_adaptation.sh` currently keeps the
first `CONFIG` argument for compatibility, but the active training parameters
come from command-line flags and environment variables. For M/L/XL, pass the
matching checkpoint and set `EXTRA_ARGS="--model_type iMF-M-2"` or the
corresponding scale.

### Required Stage-3 assets

Set these paths before training:

```bash
export SD_VAE_PATH=/path/to/sd-vae-ft-mse
export LPIPS_VGG_CKPT=assets/checkpoints/vgg.pth
export DINO_CKPT_PATH=assets/checkpoints/dino_deitsmall16_pretrain.pth
export TORCH_HOME=/path/to/torch_cache
export HF_HOME=/path/to/hf_cache
```

If you do not have a local SD-VAE directory, leave `SD_VAE_PATH` unset and
`diffusers` will try to load `stabilityai/sd-vae-ft-mse` from Hugging Face.

### Stage-3 smoke test

Run one training step before launching a full job:

```bash
NPROC_PER_NODE=1 \
EPOCHS=1 \
EVAL_EPOCHS=999 \
MAX_TRAIN_STEPS=1 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
EXTRA_ARGS="--model_type iMF-B-2 --disc_start_epoch 999" \
bash scripts/stage3_train_decoder_adaptation.sh \
  configs/stage3_decoder_adaptation/imf_b2.yaml \
  "${IMAGENET_ROOT}" \
  assets/checkpoints/imf-gar/official-pytorch/iMF-B-2.pt \
  results/stage3_smoke/imf_b2
```

Expected output:

```text
results/stage3_smoke/imf_b2/checkpoints/checkpoint-*-final.pth.tar
```

### Full B-model training

```bash
NPROC_PER_NODE=8 \
EPOCHS=10 \
BATCH_SIZE=32 \
NUM_WORKERS=8 \
EXTRA_ARGS="--model_type iMF-B-2 --minimum_noise_level 0.25 --maximum_noise_level 0.45" \
bash scripts/stage3_train_decoder_adaptation.sh \
  configs/stage3_decoder_adaptation/imf_b2.yaml \
  "${IMAGENET_ROOT}" \
  assets/checkpoints/imf-gar/official-pytorch/iMF-B-2.pt \
  results/decoder_adaptation_imf_b2_noise025045
```

For other scales, change both the checkpoint and `--model_type`:

```text
iMF-M-2  assets/checkpoints/imf-gar/official-pytorch/iMF-M-2.pt
iMF-L-2  assets/checkpoints/imf-gar/official-pytorch/iMF-L-2.pt
iMF-XL-2 assets/checkpoints/imf-gar/official-pytorch/iMF-XL-2.pt
```

### Evaluate a decoder-adapted checkpoint

```bash
NPROC_PER_NODE=8 \
NUM_SAMPLES=50000 \
BATCH_SIZE=32 \
CFG_OMEGA=8.0 \
CFG_T_MIN=0.40 \
CFG_T_MAX=0.65 \
LABEL_MAP_JSON="${LABEL_MAP_JSON}" \
OPENAI_INCEPTION_GRAPH=/path/to/classify_image_graph_def.pb \
PYTHON=/path/to/tensorflow_env/bin/python \
bash scripts/stage3_evaluate_posttrained_decoder.sh \
  iMF-B-2 \
  assets/checkpoints/imf-gar/official-pytorch/iMF-B-2.pt \
  /path/to/decoder_adapted_checkpoint.pth.tar \
  "${IMAGENET_VAL}" \
  results/eval_posttrained_imf_b2 \
  "${FID_REFERENCE_FILE}"
```

Typical outputs:

```text
results/eval_posttrained_imf_b2/Generated_CFG_iMF-B-2.npz
results/eval_posttrained_imf_b2/fid_results_openai.json
```

## Verified Project-Server Commands

These commands were smoke-tested on `loginpoint-3`.

### Server Stage-2 smoke

```bash
PYTHON=/mnt/nfs/wenjie/FD-Loss/.venv/bin/python \
PYTHONPATH=/mnt/nfs/wenjie/GAR-FID/.deps:src:third_party/ifid \
LABEL_MAP_JSON=/mnt/nfs/wenjie/IMF-GAP_V1/scripts/imagenet_val_dir_to_index.json \
FID_REFERENCE_FILE=/mnt/nfs/wenjie/IMF-GAP_V1/metric/VIRTUAL_imagenet256_labeled.npz \
PYTORCH_FID_INCEPTION_CKPT=/mnt/nfs/wenjie/torch_cache/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth \
ETA_T=0.3 NUM_SAMPLES=2 BATCH_SIZE=1 NUM_WORKERS=0 SAVE_NPZ=1 \
bash scripts/stage2_run_sit_garfid.sh \
  sdvae_b \
  /mnt/nfs/wenjie/dataset/imagenet/val \
  /mnt/nfs/wenjie/IMF-GAP_V1/metric/VIRTUAL_imagenet256_labeled.npz \
  results/stage2_runtime_check/wrapper_sdvae_b_eta03_n2
```

This produced:

```text
results/stage2_runtime_check/wrapper_sdvae_b_eta03_n2/summary.json
```

### Server Stage-3 smoke

The server environment contains `bitsandbytes`, and recent `diffusers` imports
its quantizer even though Stage 3 does not use quantization. On compute nodes
without a C compiler this can fail during import. The verified server run used
a runtime-only shim at:

```text
/mnt/nfs/wenjie/GAR-FID/.runtime_shims
```

Smoke command:

```bash
PYTHONPATH=/mnt/nfs/wenjie/GAR-FID/.runtime_shims:/mnt/nfs/wenjie/GAR-FID/.deps:/mnt/nfs/wenjie/GAR-FID/src \
LPIPS_VGG_CKPT=/mnt/nfs/wenjie/GAR-FID/src/models/cache/vgg.pth \
DINO_CKPT_PATH=/mnt/nfs/wenjie/torch_cache/hub/checkpoints/dino_deitsmall16_pretrain.pth \
TORCH_HOME=/mnt/nfs/wenjie/torch_cache \
HF_HOME=/mnt/nfs/wenjie/hf_cache \
SD_VAE_PATH=/mnt/nfs/wenjie/vae_assets/sd-vae-ft-mse \
/mnt/nfs/wenjie/FD-Loss/.venv/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=1 src/post_train.py \
  --dataset_dir /mnt/nfs/wenjie/dataset/imagenet_idx \
  --pretrained_imf_pytorch /mnt/nfs/wenjie/IMF-GAP_V1/checkpoints/pytorch/iMF-B-2.pt \
  --use_pytorch_imf \
  --checkpoint_dir results/stage3_runtime_check/checkpoints \
  --results_dir results/stage3_runtime_check/results \
  --saver_dir results/stage3_runtime_check/saver \
  --yaml_dir results/stage3_runtime_check/resolved_configs \
  --epochs 1 \
  --eval_epochs 999 \
  --batch_size 1 \
  --workers 0 \
  --max_train_steps 1 \
  --model_type iMF-B-2 \
  --saver_name_pre stage3_smoke_imf_b2 \
  --disc_start_epoch 999
```

This produced:

```text
results/stage3_runtime_check/checkpoints/checkpoint-stage3_smoke_imf_b2-final.pth.tar
```

## Troubleshooting

### `label_map_json is missing`

Build the label map:

```bash
bash scripts/prepare_imagenet_labels.sh \
  /path/to/imagenet/val \
  assets/generated/imagenet_val_dir_to_index.json
```

Then pass:

```bash
LABEL_MAP_JSON=assets/generated/imagenet_val_dir_to_index.json
```

### `PyTorch FID Inception checkpoint not found`

Set:

```bash
export PYTORCH_FID_INCEPTION_CKPT=/path/to/pt_inception-2015-12-05-6726825d.pth
```

### `args.json not found` or `Checkpoint file not found`

Download Stage-2 assets:

```bash
bash scripts/stage2_download_ifid_assets.sh --group_ids sdvae_b
```

or edit `assets/sit_checkpoint_registry_template.csv` to point to your local
SiT checkpoint directory.

### Stage 3 cannot find ImageNet train

Pass the ImageNet root that contains both `train/` and `val/`:

```bash
/path/to/imagenet
```

On the project server, use:

```bash
/mnt/nfs/wenjie/dataset/imagenet_idx
```

### `DINO checkpoint not found`

Set:

```bash
export DINO_CKPT_PATH=/path/to/dino_deitsmall16_pretrain.pth
```

### `LPIPS checkpoint not found`

Set:

```bash
export LPIPS_VGG_CKPT=/path/to/vgg.pth
```

### `diffusers` fails because `bitsandbytes` triggers Triton/C compiler setup

This is an environment issue. Stage 3 does not require bitsandbytes
quantization. Use an environment without bitsandbytes, install a working C
compiler on the compute node, or mask bitsandbytes from this run as done in
the verified server command above.

## Notes for Reproducing Paper Numbers

- Stage-2 numbers depend on the exact SiT/iFID checkpoint, tokenizer config,
  ImageNet preprocessing, and FID statistics.
- Stage-3 gFID numbers depend on the iMF checkpoint, decoder-adapted checkpoint,
  CFG parameters, and the FID backend.
- Do not commit generated `.npz` samples, `.pb` Inception graphs, pretrained
  checkpoints, or ImageNet assets. The `.gitignore` intentionally excludes
  these files.

## Third-Party Code

Stage 2 includes a minimal iFID/SiT runtime under `third_party/ifid`. This code
is kept separate from the GAR-FID code path. Check upstream licenses before a
public release and keep third-party notices up to date.

## Citation

If you use this code, please cite the GAR-FID paper. A BibTeX entry will be
added after the paper metadata is finalized.

```bibtex
@article{
}
```
