MODEL_DEFAULTS = {
    "iMF-B-2": {"omega": 8.0, "t_min": 0.4, "t_max": 0.65},
    "iMF-M-2": {"omega": 10.5, "t_min": 0.4, "t_max": 0.6},
    "iMF-L-2": {"omega": 10.5, "t_min": 0.4, "t_max": 0.6},
    "iMF-XL-2": {"omega": 8.0, "t_min": 0.42, "t_max": 0.62},
}


def model_defaults(model_type):
    try:
        return MODEL_DEFAULTS[model_type].copy()
    except KeyError as exc:
        raise ValueError(f"Unsupported iMF model: {model_type}") from exc
