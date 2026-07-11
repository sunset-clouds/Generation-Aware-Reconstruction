"""
FID/IS computation using torch-fidelity (PyTorch Inception V3).

Consistent with iMF official PyTorch evaluation using jit_in256_stats.npz.
Adapted from imeanflow-torch/utils/fidelity_wrapper.py.

Supports two input modes:
1. Directory of PNG images (default torch-fidelity usage)
2. NPZ file with 'arr_0' key (converted to temp PNG dir internally)

Usage:
    from metric.fidelity import TorchFidelityEvaluator

    evaluator = TorchFidelityEvaluator()
    fid, is_score = evaluator.compute_fid_and_is(
        sample_dir="/path/to/generated_images/",
        ref_stats="/path/to/jit_in256_stats.npz"
    )
"""

import os
import shutil
import tempfile
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

JIT_IN256_STATS_PATH = os.environ.get(
    "JIT_IN256_STATS",
    os.path.join(os.path.dirname(__file__), "jit_in256_stats.npz"),
)


def _resolve_ref_stats(path: str) -> str:
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        "torch-fidelity reference statistics not found. Set JIT_IN256_STATS "
        f"or pass --torch_fidelity_ref. Missing path: {path}"
    )


def npz_to_png_dir(npz_path: str, output_dir: Optional[str] = None) -> str:
    """
    Convert NPZ file (arr_0 key, uint8 NHWC) to a directory of PNG images.
    Returns the output directory path.
    """
    import cv2

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="fid_pngs_")
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(npz_path)
    images = data["arr_0"]
    print(f"Converting {len(images)} images from NPZ to PNG...")

    for i in tqdm(range(len(images)), desc="NPZ -> PNG"):
        img = images[i]
        if img.shape[-1] == 3:
            img = img[:, :, ::-1]  # RGB -> BGR for cv2
        cv2.imwrite(os.path.join(output_dir, f"{str(i).zfill(5)}.png"), img)

    return output_dir


class TorchFidelityEvaluator:
    """
    FID/IS evaluator using torch-fidelity (PyTorch Inception V3).
    Consistent with iMF official PyTorch branch using jit_in256_stats.npz.
    """

    def __init__(self, cuda: bool = True):
        self.cuda = cuda

    def compute_fid_and_is(
        self,
        sample_input: str,
        ref_stats: str = JIT_IN256_STATS_PATH,
    ) -> Tuple[float, float]:
        """
        Compute FID and IS using torch-fidelity.

        Args:
            sample_input: Path to directory of PNG images, or path to NPZ file.
            ref_stats: Path to reference stats NPZ (must have mu/sigma keys).

        Returns:
            Tuple of (FID, IS_mean)
        """
        from torch_fidelity.helpers import get_kwarg, vassert, vprint
        from torch_fidelity.metric_fid import (
            fid_featuresdict_to_statistics_cached,
            fid_statistics_to_metric,
        )
        from torch_fidelity.metric_isc import isc_featuresdict_to_metric
        from torch_fidelity.utils import (
            create_feature_extractor,
            extract_featuresdict_from_input_id_cached,
            get_cacheable_input_name,
        )

        temp_dir = None
        try:
            # Convert NPZ to PNG dir if needed
            if sample_input.endswith(".npz"):
                temp_dir = tempfile.mkdtemp(prefix="fid_pngs_")
                sample_dir = npz_to_png_dir(sample_input, temp_dir)
            else:
                sample_dir = sample_input

            ref_path = _resolve_ref_stats(ref_stats)

            kwargs = {
                'input1': sample_dir,
                'input2': ref_path,
                'cuda': self.cuda,
                'isc': True,
                'fid': True,
                'kid': False,
                'ppl': False,
                'verbose': True,
                'feature_extractor': 'inception-v3-compat',
                'feature_layer_isc': 'logits_unbiased',
                'feature_layer_fid': '2048',
                'batch_size': 64,
                'samples_find_deep': False,
                'samples_find_ext': 'png,jpg,jpeg',
                'samples_ext_lossy': 'jpg,jpeg',
                'samples_shuffle': True,
                'rng_seed': 2020,
                'save_cpu_ram': False,
                'cache': True,
                'cache_root': os.path.join(tempfile.gettempdir(), "fidelity_cache"),
                'input1_cache_name': None,
                'input1_model_z_type': 'normal',
                'input1_model_z_size': None,
                'input1_model_num_classes': 0,
                'input1_model_num_samples': None,
                'input2_cache_name': None,
                'input2_model_z_type': 'normal',
                'input2_model_z_size': None,
                'input2_model_num_classes': 0,
                'input2_model_num_samples': None,
                'datasets_root': None,
                'datasets_download': True,
                'feature_extractor_weights_path': None,
                'isc_splits': 10,
                'kid_subsets': 100,
                'kid_subset_size': 1000,
                'kid_degree': 3,
                'kid_gamma': None,
                'kid_coef0': 1.0,
                'ppl_epsilon': 1e-4,
                'ppl_reduction': 'mean',
                'ppl_sample_similarity': 'lpips-vgg16',
                'ppl_sample_similarity_resize': 64,
                'ppl_sample_similarity_dtype': 'uint8',
                'ppl_discard_percentile_lower': 1,
                'ppl_discard_percentile_higher': 99,
                'ppl_z_interp_mode': 'lerp',
            }

            feature_layers = set()
            feature_layer_isc = kwargs['feature_layer_isc']
            feature_layer_fid = kwargs['feature_layer_fid']
            feature_layers.add(feature_layer_isc)
            feature_layers.add(feature_layer_fid)

            feat_extractor = create_feature_extractor(
                kwargs['feature_extractor'], list(feature_layers), **kwargs
            )

            vprint(True, "Extracting features from generated images...")
            featuresdict_1 = extract_featuresdict_from_input_id_cached(
                1, feat_extractor, **kwargs
            )

            # IS
            metric_isc = isc_featuresdict_to_metric(
                featuresdict_1, feature_layer_isc, **kwargs
            )

            # FID
            cacheable_input1_name = get_cacheable_input_name(1, **kwargs)
            print("Computing FID stats for generated images...")
            fid_stats_1 = fid_featuresdict_to_statistics_cached(
                featuresdict_1,
                cacheable_input1_name,
                feat_extractor,
                feature_layer_fid,
                **kwargs,
            )

            x = np.load(ref_path)
            if "mu" in x.files:
                # Reference is a pre-computed stats file (like jit_in256_stats.npz)
                fid_stats_2 = {"mu": x["mu"], "sigma": x["sigma"]}
            else:
                # Reference is a raw images file (like Input.npz)
                print("Computing FID stats for reference images...")
                # We need to extract features for the reference images
                # Since ref_path is an NPZ file with 'arr_0', we need to convert it to PNGs first
                temp_ref_dir = tempfile.mkdtemp(prefix="fid_ref_pngs_")
                try:
                    ref_dir = npz_to_png_dir(ref_path, temp_ref_dir)
                    kwargs_ref = kwargs.copy()
                    kwargs_ref['input1'] = ref_dir
                    
                    featuresdict_2 = extract_featuresdict_from_input_id_cached(
                        1, feat_extractor, **kwargs_ref
                    )
                    cacheable_input2_name = get_cacheable_input_name(1, **kwargs_ref)
                    fid_stats_2 = fid_featuresdict_to_statistics_cached(
                        featuresdict_2,
                        cacheable_input2_name,
                        feat_extractor,
                        feature_layer_fid,
                        **kwargs_ref,
                    )
                finally:
                    if os.path.exists(temp_ref_dir):
                        shutil.rmtree(temp_ref_dir)

            metric_fid = fid_statistics_to_metric(fid_stats_1, fid_stats_2, True)

            fid = metric_fid["frechet_inception_distance"]
            is_mean = metric_isc["inception_score_mean"]

            return float(fid), float(is_mean)

        finally:
            if temp_dir is not None and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def compute_from_image_dir(
        self,
        image_dir: str,
        ref_stats: str = JIT_IN256_STATS_PATH,
    ) -> dict:
        """
        Compute FID and IS from a directory of images.
        Returns dict with 'fid' and 'is_mean' keys.
        """
        fid, is_mean = self.compute_fid_and_is(image_dir, ref_stats)
        return {"fid": fid, "is_mean": is_mean}

    def close(self):
        pass
