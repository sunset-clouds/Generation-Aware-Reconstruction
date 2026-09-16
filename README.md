<div align="center">

# Bridging Reconstruction and Generation

### A Latent Distribution Perspective on Evaluation and Improvement

[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](docs/assets/GAR_paper.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-3b6ea8.svg)](https://sunset-clouds.github.io/Generation-Aware-Reconstruction/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](docs/assets/GAR_paper.pdf)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-ffd21e.svg)](https://huggingface.co/sunset-clouds/GAR/tree/main)

**[Xianghong Fang](https://sunset-clouds.github.io/)<sup>1,*</sup> · Wenjie Shu<sup>1,*</sup> · Tongda Xu<sup>2</sup> · Wenlong Mou<sup>1</sup> · Dehan Kong<sup>1</sup> · Tim G. J. Rudner<sup>1,3</sup>**

<sup>1</sup>University of Toronto &nbsp;&nbsp; <sup>2</sup>Independent &nbsp;&nbsp; <sup>3</sup>Vijil

<sup>*</sup>Equal contribution

</div>

> **TL;DR:** Reconstruction and generation apply the same decoder to different latent distributions. Generation-aware reconstruction (GAR) probes the transition between them, provides an evaluation metric that correlates strongly with generative performance, and enables decoder adaptation that improves generation without additional inference cost.

<p align="center">
  <a href="docs/assets/Figure1.pdf">
    <img src="docs/assets/Figure1.png" width="100%" alt="Reconstruction-generation latent distribution mismatch and generation-aware reconstruction">
  </a>
  <br>
  <big><big>Bridging reconstruction and generation through latent distributions.</big></big>
</p>

## Overview

Why can strong reconstruction coexist with weak generation? Reconstruction decodes encoder latents drawn from $P_e$, while generation decodes latents drawn from $P_g$. Reconstruction FID (rFID) therefore evaluates the decoder under a different input distribution from generative FID (gFID).

**Generation-aware reconstruction (GAR)** makes this transition observable. It perturbs encoder latents, denoises them with a frozen generative model, and decodes the resulting intermediate latents. The noise level $\eta_t$ controls the trajectory: $\eta_t=0$ recovers standard reconstruction, while $\eta_t=1$ recovers generation from pure noise.

- **Evaluation:** GAR-FID measures FID between decoded GAR outputs and the source image set, tracking decoder behavior as its inputs approach the generation-time distribution.
- **Improvement:** Intermediate GAR latents retain correspondence with source images and provide paired supervision for decoder adaptation. Only the decoder is updated; the encoder and generative model remain frozen.

<p align="center">
  <a href="docs/assets/GAR_pipeline.pdf">
    <img src="docs/assets/GAR_pipeline.png" width="100%" alt="GAR pipeline: encode, perturb, denoise with a frozen generator, and decode">
  </a>
  <br>
  <sub>The generation-aware reconstruction pipeline.</sub>
</p>

## Main Results

### Correlation with generative performance

Across 13 VAEs and two SiT generator scales, GAR-FID correlates more strongly with gFID as the trajectory approaches generation. The table below reports Spearman rank correlation (SRCC; higher is better) at $\eta_t=0.8$ for GAR-FID. The rFID and iFID baselines are our reproduced results.

| Generator setting | CFG | rFID | iFID | GAR-FID ($\eta_t=0.8$) |
|:--|:--:|--:|--:|--:|
| SiT-B | No | −0.36 | 0.83 | **0.96** |
| SiT-B | Yes | −0.30 | 0.79 | **0.94** |
| SiT-XL | No | −0.34 | 0.88 | **0.99** |
| SiT-XL | Yes | −0.39 | 0.90 | **0.96** |
| SiT-mixed | No | −0.30 | 0.77 | **0.99** |
| SiT-mixed | Yes | −0.28 | 0.70 | **0.95** |

CFG denotes classifier-free guidance. GAR-FID provides a diagnostic along the latent trajectory; direct gFID evaluation remains necessary when final sample quality is the target. See the [paper](docs/assets/GAR_paper.pdf) for Pearson correlations and the full noise-level sweep.

### Decoder adaptation improves generation

Decoder adaptation (DA) improves gFID at every tested iMF scale under both the official iMF and OpenAI evaluation protocols. Below are our reproduced results under the **OpenAI protocol**; lower gFID is better.

| Model | Without DA, no CFG | With DA, no CFG | Without DA, with CFG | With DA, with CFG |
|:--|--:|--:|--:|--:|
| iMF-B/2 | 16.41 | **12.41** | 3.47 | **3.12** |
| iMF-M/2 | 11.92 | **9.66** | 2.40 | **2.30** |
| iMF-L/2 | 9.26 | **7.41** | 1.90 | **1.80** |
| iMF-XL/2 | 9.41 | **7.61** | 1.80 | **1.73** |

Adaptation changes only the decoder and adds no inference cost. Under the official iMF protocol, adapted iMF-XL/2 achieves a gFID of **1.56 with CFG**; results from the two protocols should be compared within their respective settings.

<p align="center">
  <a href="docs/assets/class_014_indigo_bunting.pdf">
    <img src="docs/assets/class_014_indigo_bunting.png" width="100%" alt="Indigo bunting generation before and after decoder adaptation using identical generative latents">
  </a>
  <br>
  <sub>Identical generative latents decoded before (top) and after (bottom) adaptation.</sub>
</p>

### Generation awareness and source correspondence

<p align="center">
  <a href="docs/assets/GAR_examples.pdf">
    <img src="docs/assets/GAR_examples.png" width="100%" alt="GAR outputs at increasing noise levels showing the progressive loss of source structure">
  </a>
  <br>
  <sub>GAR outputs from reconstruction toward generation as the noise level increases.</sub>
</p>

Increasing the noise level moves GAR latents toward the generation-time distribution while progressively weakening correspondence with the source image. Intermediate noise levels balance generation awareness with the paired supervision needed for decoder adaptation.

## Repository Structure

The released code covers GAR-FID evaluation with SiT/iFID checkpoints (**Stage 2**) and decoder adaptation for iMF (**Stage 3**).

```text
assets/
  sit_checkpoint_registry_template.csv  # SiT/iFID checkpoint registry
docs/
  assets/                               # Paper and figures
  index.html                            # Project website
scripts/
  prepare_imagenet_labels.sh             # Validation-folder label mapping
  stage2_*.sh                           # GAR-FID evaluation and correlations
  stage3_*.sh                           # Decoder training and evaluation
src/
  integrations/sit/                     # SiT checkpoint loading and GAR adapters
  pipelines/                            # Stage-2 evaluation pipelines
  post_train.py                         # Decoder-adaptation training
  models/                               # Tokenizer, iMF, and loss implementations
  data/, metric/, utils/                # Shared runtime utilities
third_party/ifid/                       # Minimal iFID/SiT runtime
requirements.txt                       # PyTorch runtime dependencies
requirements-fid.txt                   # Separate TensorFlow FID environment
requirements-jax-conversion.txt        # Optional legacy checkpoint conversion
```

## Setup

The commands below use Bash and Python 3.10 or 3.11 with CUDA-enabled PyTorch. Run them from the repository root.

```bash
git clone https://github.com/sunset-clouds/Generation-Aware-Reconstruction.git
cd Generation-Aware-Reconstruction

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install a CUDA-compatible PyTorch build first if needed.
pip install -r requirements.txt
pip install -r third_party/ifid/requirements.txt
export PYTHONPATH=src:third_party/ifid:${PYTHONPATH:-}
```

Stage 3 runs in PyTorch. JAX/Flax is only needed for optional conversion of legacy checkpoints, using `requirements-jax-conversion.txt` and [`scripts/stage3_convert_imf_checkpoint.sh`](scripts/stage3_convert_imf_checkpoint.sh).

### Checkpoints

GAR checkpoints are hosted on [Hugging Face](https://huggingface.co/sunset-clouds/GAR/tree/main). Download the repository with:

```bash
hf download sunset-clouds/GAR --local-dir assets/checkpoints/GAR
```

Set `IMF_CKPT` to the pretrained iMF PyTorch checkpoint and `DECODER_CKPT` to the adapted decoder checkpoint selected for evaluation. Both must match the model scale; the examples below use iMF-B/2.

```bash
export IMF_CKPT=/path/to/iMF-B-2.pt
export DECODER_CKPT=/path/to/decoder_adapted_checkpoint.pth.tar
```

For Stage 2, download the SiT/iFID assets listed in the [checkpoint registry](assets/sit_checkpoint_registry_template.csv):

```bash
# Download the SD-VAE / SiT-B example used below.
bash scripts/stage2_download_ifid_assets.sh --group_ids sdvae_b

# Or download all registry entries marked for execution.
bash scripts/stage2_download_ifid_assets.sh --only_should_run
```

### Data and evaluation assets

Prepare ImageNet at 256 × 256 resolution, with `train/` and `val/` folders, and configure the required local paths:

```bash
export IMAGENET_ROOT=/path/to/imagenet
export IMAGENET_VAL=${IMAGENET_ROOT}/val
export LABEL_MAP_JSON=assets/generated/imagenet_val_dir_to_index.json
export FID_REFERENCE_FILE=/path/to/VIRTUAL_imagenet256_labeled.npz
export PYTORCH_FID_INCEPTION_CKPT=/path/to/pt_inception-2015-12-05-6726825d.pth

# Required for decoder-adaptation training.
export SD_VAE_PATH=/path/to/sd-vae-ft-mse
export LPIPS_VGG_CKPT=/path/to/vgg.pth
export DINO_CKPT_PATH=/path/to/dino_deitsmall16_pretrain.pth

bash scripts/prepare_imagenet_labels.sh \
  "${IMAGENET_VAL}" "${LABEL_MAP_JSON}"
```

ImageNet, reference statistics, and external evaluation/loss checkpoints must be supplied separately. Match the dataset preprocessing, model checkpoints, FID statistics, and evaluation protocol when reproducing the reported results.

## Running the Experiments

### GAR-FID evaluation

Run GAR-FID on the SD-VAE / SiT-B checkpoint at a single noise level:

```bash
ETA_T=0.8 NUM_SAMPLES=50000 BATCH_SIZE=16 SAVE_NPZ=1 \
bash scripts/stage2_run_sit_garfid.sh \
  sdvae_b \
  "${IMAGENET_VAL}" \
  "${FID_REFERENCE_FILE}" \
  results/stage2_sit_garfid/sdvae_b_eta08
```

For a small runtime check, set `NUM_SAMPLES=8` and `BATCH_SIZE=2`; use the full evaluation sample count for reported metrics. To evaluate the trajectory:

```bash
ETA_LIST="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0" \
NUM_SAMPLES=50000 BATCH_SIZE=16 SAVE_NPZ=1 \
bash scripts/stage2_run_sit_garfid_sweep.sh \
  sdvae_b \
  "${IMAGENET_VAL}" \
  "${FID_REFERENCE_FILE}" \
  results/stage2_sit_garfid/sdvae_b
```

Use the registry to select other checkpoint groups. [`scripts/stage2_compute_correlations.sh`](scripts/stage2_compute_correlations.sh) computes Pearson and Spearman correlations from a collected results CSV.

### Decoder adaptation

Train the decoder with a frozen iMF-B/2 generator and noise levels sampled from $[0.25, 0.45]$:

```bash
NPROC_PER_NODE=8 EPOCHS=10 BATCH_SIZE=32 NUM_WORKERS=8 \
EXTRA_ARGS="--model_type iMF-B-2 --minimum_noise_level 0.25 --maximum_noise_level 0.45" \
bash scripts/stage3_train_decoder_adaptation.sh \
  "${IMAGENET_ROOT}" \
  "${IMF_CKPT}" \
  results/decoder_adaptation_imf_b2_noise025045
```

For other scales, change both `IMF_CKPT` and `--model_type` to the corresponding `iMF-M-2`, `iMF-L-2`, or `iMF-XL-2`. A one-step training check can use `NPROC_PER_NODE=1`, `BATCH_SIZE=1`, `NUM_WORKERS=0`, `MAX_TRAIN_STEPS=1`, and `EVAL_EPOCHS=999`.

### Evaluate the adapted decoder

The evaluation wrapper generates samples with PyTorch and computes FID with the OpenAI TensorFlow backend. Create a separate FID environment once:

```bash
python -m venv .venv-fid
.venv-fid/bin/python -m pip install -r requirements-fid.txt
export OPENAI_INCEPTION_GRAPH=/path/to/classify_image_graph_def.pb
```

Keep the PyTorch environment active for `torchrun`; `PYTHON` below selects the separate interpreter for the final FID step. Set `DECODER_CKPT` to a downloaded adapted decoder or the final checkpoint from your training run.

```bash
NPROC_PER_NODE=8 NUM_SAMPLES=50000 BATCH_SIZE=32 \
CFG_OMEGA=8.0 CFG_T_MIN=0.40 CFG_T_MAX=0.65 \
PYTHON="${PWD}/.venv-fid/bin/python" \
bash scripts/stage3_evaluate_posttrained_decoder.sh \
  iMF-B-2 \
  "${IMF_CKPT}" \
  "${DECODER_CKPT}" \
  "${IMAGENET_VAL}" \
  results/eval_posttrained_imf_b2 \
  "${FID_REFERENCE_FILE}"
```

This example evaluates generation with CFG. Use the model-specific guidance settings for each scale when reproducing the paper's results.

## Evaluation and Outputs

- **GAR-FID:** each single-point output directory contains `summary.json` and `preview_grid.png`, with samples saved when `SAVE_NPZ` is enabled.
- **Decoder adaptation:** the output root contains `checkpoints/`, `results/`, `saver/`, and `resolved_configs/`.
- **Generation evaluation:** outputs include `Generated_CFG_iMF-B-2.npz` and `fid_results_openai.json` for the B/2 example above.
- **Reproducibility:** launch recipes are available under [`scripts/`](scripts/). Update local dataset, checkpoint, statistics, and output paths before execution.

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{fang2026bridging,
  title   = {Bridging Reconstruction and Generation: A Latent Distribution Perspective on Evaluation and Improvement},
  author  = {Fang, Xianghong and Shu, Wenjie and Xu, Tongda and Mou, Wenlong and Kong, Dehan and Rudner, Tim G. J.},
  journal = {Arxiv},
  year    = {2026}
}
```

## Acknowledgements

This work builds on iMF, SiT, iFID, and SD-VAE. We thank their authors and the contributors to the perceptual-metric and evaluation libraries used in this repository. The bundled iFID/SiT runtime is located in [`third_party/ifid/`](third_party/ifid/); see [Third-Party Notices](THIRD_PARTY_NOTICES.md).
