<div align="center">

# A Unified Rate–Distortion Perspective

### on Vector, Product, and Scalar Quantization

[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](docs/assets/Unified_Rate_Distortion_Perspective.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-3b6ea8.svg)](https://vq-research.github.io/Rate-Distortion-Perspective/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://vq-research.github.io/Rate-Distortion-Perspective/assets/Unified_Rate_Distortion_Perspective.pdf)

**[Xianghong Fang](https://sunset-clouds.github.io/)<sup>1</sup> · Wenlong Mou<sup>1</sup> · Yuan Yuan<sup>2</sup> · Dehan Kong<sup>1</sup> · Tim G. J. Rudner<sup>1,3</sup>**

<sup>1</sup>University of Toronto &nbsp;&nbsp; <sup>2</sup>Boston College &nbsp;&nbsp; <sup>3</sup>Vijil

</div>

> **TL;DR:** We formulate discrete quantization as fixed-rate lossy compression, identify distortion minimization as the fundamental optimization objective, establish the conditions required for fair intrinsic comparison, and show that modern VQ methods achieve lower distortion than PQ and SQ under matched latent distributions and coding rates.

<p align="center">
  <a href="docs/assets/Figure1.pdf">
    <img src="docs/assets/Figure1.png" width="100%" alt="Overview of the unified rate-distortion perspective">
  </a>
  <br>
  <big><big>Overview of the unified rate–distortion perspective.</big></big>
</p>

## Overview

We compare VQ, PQ, and SQ under matched latent distributions and coding rates to identify what quantizers should optimize and which family minimizes distortion.

## Main Results

All methods within an experimental setting use the same source representation and code-space cardinality. Latent-space experiments use (T=512) and (K=65536). Pixel-space experiments use (T=4096) and (K=65536). Each metric below reports the best result within a quantizer family; the method attaining the lowest distortion may differ from the method attaining the lowest rFID.

### Best controlled results

| Space | Dataset | Family | Best distortion method | Best rFID method | Distortion ↓ | rFID ↓ |
|:--|:--|:--:|:--|:--|--:|--:|
| Latent | ImageNet-1K | **VQ** | MMD VQ | MMD VQ | **0.201** | **0.86** |
| Latent | ImageNet-1K | PQ | EMA VP2 | EMA VP2 | 0.209 | 0.93 |
| Latent | ImageNet-1K | SQ | BSQ | BSQ | 0.231 | 1.07 |
| Latent | FFHQ | **VQ** | MMD VQ | MMD VQ | **0.131** | **0.85** |
| Latent | FFHQ | PQ | EMA VP2 | EMA VP2 | 0.136 | 1.05 |
| Latent | FFHQ | SQ | BSQ | BSQ | 0.155 | 1.54 |
| Latent | CelebA-HQ | **VQ** | Wasserstein VQ / MMD VQ | Wasserstein VQ | **0.111** | **1.73** |
| Latent | CelebA-HQ | PQ | EMA VP2 | EMA VP2 | 0.115 | 1.96 |
| Latent | CelebA-HQ | SQ | BSQ | FSQ | 0.133 | 2.24 |
| Pixel | CelebA-HQ | **VQ** | MMD VQ | MMD VQ | **0.0021** | **3.64** |
| Pixel | CelebA-HQ | PQ | Online VP2 / MMD VP2 | MMD VP2 | 0.0023 | 3.77 |
| Pixel | CelebA-HQ | SQ | FSQ | FSQ | 0.0032 | 4.54 |

Latent-space rFID is measured after decoder adaptation. Pixel-space models use single-stage training.

### Rate–distortion curves

<p align="center">
  <a href="docs/assets/RD_curve.pdf">
    <img src="docs/assets/RD_curve.png" width="62%" alt="Rate-distortion curves on ImageNet-1K">
  </a>
  <br>
  <sub>Rate–distortion curves on ImageNet-1K.</sub>
</p>

Increasing the per-token coding rate reduces distortion for all three quantization families. VQ achieves the lowest distortion at every evaluated operating point, followed by PQ and SQ.

### Distortion and reconstruction fidelity

Spearman rank correlations show that distortion tracks rFID more closely than codebook utilization across representation spaces and datasets.

| Space | Dataset | Distortion vs. rFID | Utilization vs. rFID |
|:--|:--|:--|:--|
| Latent | ImageNet-1K | &rho; = 0.996, p &lt; 10<sup>&minus;8</sup> | &rho; = &minus;0.540, p = 0.057 |
| Latent | FFHQ | &rho; = 0.979, p = 5.49 &times; 10<sup>&minus;9</sup> | &rho; = &minus;0.492, p = 0.088 |
| Latent | CelebA-HQ | &rho; = 0.944, p = 1.29 &times; 10<sup>&minus;6</sup> | &rho; = &minus;0.559, p = 0.047 |
| Pixel | CelebA-HQ | &rho; = 0.947, p = 2.91 &times; 10<sup>&minus;6</sup> | &rho; = &minus;0.413, p = 0.182 |

## Repository Structure

The repository provides two controlled experimental settings:

- [`Pixel-Space/`](Pixel-Space/) compares VQ, PQ, and SQ directly in pixel space.
- [`VQ-Transplant/`](VQ-Transplant/) compares VQ, PQ, and SQ in the latent space of a pretrained tokenizer through transplant experiments.

Both experimental directories follow a similar structure:

```text
Pixel-Space/ or VQ-Transplant/
├── scripts/    # Training, evaluation, and job scripts
├── results/    # Experimental results and metric outputs
├── logs/       # Training logs, where provided
├── record/     # Transplant training records, where provided
├── data/       # Dataset utilities and configuration
├── metric/     # Evaluation code
├── models/     # Quantizer and model implementations
└── utils/      # Supporting utilities
```

The materials under `scripts/`, `results/`, `logs/`, and `record/` make the experiments transparent and convenient to reproduce:

- **Run instructions:** shell scripts preserve the commands and configurations used for training and evaluation.
- **Reported results:** CSV files and metric logs contain the experimental results used in our analysis.
- **Training records:** log and record files preserve the training process for inspecting individual runs.

## Setup

Create a Python environment with a CUDA-enabled PyTorch installation, then install the dependencies used by the selected experiment:

```bash
git clone https://github.com/VQ-Research/Rate-Distortion-Perspective.git
cd Rate-Distortion-Perspective

conda create -n rate-distortion python=3.10 -y
conda activate rate-distortion

# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install torchvision timm einops omegaconf safetensors scipy pandas pillow \
  opencv-python tqdm piq pyiqa ruamel.yaml clean-fid \
  pytorch-image-generation-metrics
```

All experiments use images resized to 256 &times; 256. Dataset paths and pretrained-tokenizer paths are configured in the corresponding launch scripts.

## Running the Experiments

Ready-to-edit launch recipes are provided in each implementation's `scripts/` directory.

### Pixel-space comparison

The [`Pixel-Space/scripts`](Pixel-Space/scripts/) directory contains configurations for Vanilla VQ, EMA VQ, Online VQ, Wasserstein VQ, MMD VQ, their PQ counterparts, FSQ, LFQ, and BSQ. For example:

```bash
cd Pixel-Space
bash scripts/mmd_vq_celeba.sh
```

Pixel-space experiments use PixelUnshuffle/PixelShuffle with shared convolutional projectors, (T=4096) tokens, and (K=65536).

### Latent-space transplant comparison

The [`VQ-Transplant/scripts`](VQ-Transplant/scripts/) directory contains substitution, decoder-adaptation, and rFID scripts for ImageNet-1K, FFHQ, and CelebA-HQ. For example:

```bash
cd VQ-Transplant
bash scripts/transplant/ImageNet/mmd_pq.sh
```

Latent-space experiments share the same pretrained VAR encoder output and use (T=512) tokens with (K=65536). Quantizer substitution freezes the pretrained encoder and decoder; decoder adaptation then freezes the encoder and transplanted quantizer while updating the decoder.

## Evaluation and Logs

- Reconstruction evaluation reports quantization distortion, codebook utilization, PSNR, SSIM, LPIPS, rFID, and rIS where applicable.
- Pixel-space outputs are available under [`Pixel-Space/results`](Pixel-Space/results/) and [`Pixel-Space/logs`](Pixel-Space/logs/).
- Latent-space outputs are available under [`VQ-Transplant/results`](VQ-Transplant/results/) and [`VQ-Transplant/record`](VQ-Transplant/record/).
- rFID launch scripts are provided under the corresponding `scripts/rFID/` directories.

Some scripts preserve the original cluster paths. Update dataset, checkpoint, sample, statistics, and output paths before execution.

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{Fang2026ratedistortion,
  title   = {A Unified Rate--Distortion Perspective on Vector,
             Product, and Scalar Quantization},
  author  = {Fang, Xianghong and Mou, Wenlong and Yuan, Yuan and
             Kong, Dehan and Rudner, Tim G. J.},
  journal = {arXiv},
  year    = {2026}
}
```

## Acknowledgements

The latent-space experiments build on the pretrained [VAR tokenizer](https://github.com/FoundationVision/VAR) and the [VQ-Transplant](https://github.com/VQ-Research/VQ-Transplant) framework. We thank the authors of the quantization, tokenizer, perceptual-metric, and evaluation libraries used in this repository.
