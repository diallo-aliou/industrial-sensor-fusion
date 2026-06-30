import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.loader import load_raw
from src.features.freq_domain import compute_freq_features
from src.features.time_domain import compute_time_features, compute_trend_features

_FS_HIGH = 100.0
_FS_LOW = 1.0


class ModalityFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = config_path
        self._high_rate: list[str] = []
        self._low_rate: list[str] = []
        self._all_sensors: list[str] = []
        self._top_k: int = 20

    def fit(self, X=None, y=None):
        config_path = Path(self.config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self._high_rate  = config["sensors"]["high_rate"]
        self._low_rate   = config["sensors"]["low_rate"]
        self._all_sensors = (
            config["sensors"]["high_rate"]
            + config["sensors"]["mid_rate"]
            + config["sensors"]["low_rate"]
        )
        self._top_k = config["features"]["freq_domain_top_k"]
        return self

    def transform(self, sensors: dict[str, np.ndarray]) -> pd.DataFrame:
        rows = []
        n_cycles = next(iter(sensors.values())).shape[0]

        for cycle_idx in range(n_cycles):
            row: dict[str, float] = {}
            for sensor_name in self._all_sensors:
                signal = sensors[sensor_name][cycle_idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    time_feats = compute_time_features(signal)
                for feat_name, val in time_feats.items():
                    row[f"{sensor_name}_{feat_name}"] = val

                if sensor_name in self._high_rate:
                    freq_feats = compute_freq_features(signal, fs=_FS_HIGH, top_k=self._top_k)
                    for feat_name, val in freq_feats.items():
                        row[f"{sensor_name}_{feat_name}"] = val

                if sensor_name in self._low_rate:
                    trend_feats = compute_trend_features(signal, fs=_FS_LOW)
                    for feat_name, val in trend_feats.items():
                        row[f"{sensor_name}_{feat_name}"] = val

            rows.append(row)

            if (cycle_idx + 1) % 500 == 0 or cycle_idx == n_cycles - 1:
                print(f"  Processed {cycle_idx + 1}/{n_cycles} cycles")

        return pd.DataFrame(rows)


def build_feature_matrix(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_path = (config_path.parent.parent / config["data"]["features_file"]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading raw sensors...")
    sensors = load_raw(str(config_path))

    print("Extracting features...")
    extractor = ModalityFeatureExtractor(config_path=str(config_path))
    extractor.fit()
    df = extractor.transform(sensors)

    df.to_parquet(out_path, index=False)
    print(f"Feature matrix saved: {df.shape} -> {out_path}")
    return df
