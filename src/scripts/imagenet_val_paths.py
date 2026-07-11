"""List ImageNet-style val image paths without importing models (for label audits)."""
import os


def list_imagenet_val_image_paths(root_dir, num_samples=50000):
    """
    Same ordering as ``ImageNetDataset`` in generate_eval_images (sorted class subdirs, sorted files).
    """
    paths = []
    image_extensions = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}
    if not os.path.isdir(root_dir):
        return paths
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if subdirs:
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(root_dir, subdir)
            for f in sorted(os.listdir(subdir_path)):
                if os.path.splitext(f)[1] in image_extensions:
                    paths.append(os.path.join(subdir_path, f))
    else:
        for f in sorted(os.listdir(root_dir)):
            if os.path.splitext(f)[1] in image_extensions:
                paths.append(os.path.join(root_dir, f))
    if num_samples is not None:
        paths = paths[:num_samples]
    return paths
