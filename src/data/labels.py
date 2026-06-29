from pathlib import Path

import numpy as np
import pandas as pd
import yaml


_FAULT_COLUMNS = ["cooler", "valve", "pump", "accumulator"]
_STABLE_COL = 4
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_config(config_path: str | Path) -> Path:
    # Anchor a relative config path to the project root so these functions work from
    # any directory (e.g. a notebook in notebooks/). Absolute paths pass through.
    p = Path(config_path)
    return p if p.is_absolute() else (_PROJECT_ROOT / p)


def load_labels(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config_path = _resolve_config(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_dir = (config_path.parent.parent / config["data"]["raw_dir"]).resolve()
    profile = np.loadtxt(raw_dir / "profile.txt", dtype=np.int32)

    labels = pd.DataFrame(profile[:, :4], columns=_FAULT_COLUMNS)
    print(f"Labels loaded: {labels.shape} — {labels.nunique().to_dict()} unique values per component")
    return labels


def load_stable_mask(config_path: str = "configs/config.yaml") -> pd.Series:
    config_path = _resolve_config(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_dir = (config_path.parent.parent / config["data"]["raw_dir"]).resolve()
    profile = np.loadtxt(raw_dir / "profile.txt", dtype=np.int32)

    mask = pd.Series(profile[:, _STABLE_COL].astype(bool), name="stable")
    print(f"Stable cycles: {mask.sum()} / {len(mask)} ({mask.mean():.1%})")
    return mask


def get_label_map(config_path: str = "configs/config.yaml") -> dict:
    config_path = _resolve_config(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return {
        component: config["labels"][component]
        for component in _FAULT_COLUMNS
    }
