from pathlib import Path

import numpy as np
import pandas as pd
import yaml


_FAULT_COLUMNS = ["cooler", "valve", "pump", "accumulator"]
_STABLE_COL = 4


def load_labels(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["data"]["raw_dir"])
    profile = np.loadtxt(raw_dir / "profile.txt", dtype=np.int32)

    labels = pd.DataFrame(profile[:, :4], columns=_FAULT_COLUMNS)
    print(f"Labels loaded: {labels.shape} — {labels.nunique().to_dict()} unique values per component")
    return labels


def load_stable_mask(config_path: str = "configs/config.yaml") -> pd.Series:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["data"]["raw_dir"])
    profile = np.loadtxt(raw_dir / "profile.txt", dtype=np.int32)

    mask = pd.Series(profile[:, _STABLE_COL].astype(bool), name="stable")
    print(f"Stable cycles: {mask.sum()} / {len(mask)} ({mask.mean():.1%})")
    return mask


def get_label_map(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return {
        component: config["labels"][component]
        for component in _FAULT_COLUMNS
    }
