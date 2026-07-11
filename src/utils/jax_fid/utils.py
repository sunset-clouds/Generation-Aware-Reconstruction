import os
import tempfile


def download(path, ckpt_dir=None):
    if os.path.exists(path):
        return path
    if ckpt_dir is None:
        ckpt_dir = os.path.join(tempfile.gettempdir(), "jax_fid")
    candidate = os.path.join(ckpt_dir, os.path.basename(path))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        "JAX FID checkpoint not found. Set JAX_FID_INCEPTION_CKPT or pass "
        f"a local checkpoint path. Missing path: {path}"
    )


def get(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]
