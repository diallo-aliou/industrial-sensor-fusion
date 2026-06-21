import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml


_UCI_URL = (
    "https://archive.ics.uci.edu/static/public/447/"
    "condition+monitoring+of+hydraulic+systems.zip"
)

_ALL_SENSORS = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1",
    "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1",
    "SE", "CE", "CP",
]


def download_data(raw_dir: str = "data/raw") -> None:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    if (raw_path / "PS1.txt").exists():
        print("Data already downloaded. Skipping.")
        return

    zip_path = raw_path / "hydraulic.zip"
    print("Downloading dataset from UCI...")
    urllib.request.urlretrieve(_UCI_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_path)

    zip_path.unlink()
    print(f"Done. Files saved to {raw_path}/")


def _parse_sensor_file(filepath: Path) -> np.ndarray:
    return np.loadtxt(filepath, dtype=np.float32)


def load_raw(config_path: str = "configs/config.yaml") -> dict[str, np.ndarray]:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["data"]["raw_dir"])

    sensors: dict[str, np.ndarray] = {}
    for name in _ALL_SENSORS:
        filepath = raw_dir / f"{name}.txt"
        arr = _parse_sensor_file(filepath)
        sensors[name] = arr
        print(f"  {name:<6} loaded: {arr.shape}")

    print(f"\nAll {len(sensors)} sensors loaded.")
    return sensors
