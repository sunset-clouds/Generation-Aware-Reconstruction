"""Resolve existing SiT assets across the public-directory rename."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_sit_asset_path(path_value: str) -> str:
    """Prefer the requested path; reuse assets/stage2_assets when assets/sit is absent."""
    if not path_value:
        return path_value
    candidate = Path(path_value).expanduser()
    absolute = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    if absolute.exists():
        return path_value
    parts = absolute.parts
    for index in range(len(parts) - 1):
        if parts[index:index + 2] == ("assets", "sit"):
            legacy = Path(*parts[:index + 1], "stage2_assets", *parts[index + 2:])
            if legacy.exists():
                return str(legacy.resolve())
            break
    return path_value
