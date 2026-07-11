"""
Metric utilities for FID, IS, PSNR, SSIM, LPIPS evaluation.

Two FID evaluation backends:
- OpenAI TF Inception (FIDEvaluator): consistent with ADM/DiT/SiT benchmarks
- torch-fidelity PyTorch Inception (TorchFidelityEvaluator): consistent with iMF official

Use get_evaluator() factory to select backend by name.
"""

__all__ = ["get_evaluator", "PSNR", "SSIM", "LPIPS"]


def __getattr__(name):
    if name in {"PSNR", "SSIM", "LPIPS"}:
        from .metric import LPIPS, PSNR, SSIM

        return {"PSNR": PSNR, "SSIM": SSIM, "LPIPS": LPIPS}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_evaluator(backend="openai"):
    """
    Factory to get a FID evaluator by backend name.

    Args:
        backend: "openai" for TF Inception (VIRTUAL_imagenet256_labeled.npz),
                 "torch_fidelity" for PyTorch Inception (jit_in256_stats.npz).

    Returns:
        Evaluator instance with compute_fid_and_is(ref, sample) and close() methods.
    """
    if backend == "openai":
        from .evaluator import FIDEvaluator
        return FIDEvaluator()
    elif backend == "torch_fidelity":
        from .fidelity import TorchFidelityEvaluator
        return TorchFidelityEvaluator()
    else:
        raise ValueError(f"Unknown evaluator backend: {backend}. "
                         f"Choose 'openai' or 'torch_fidelity'.")
