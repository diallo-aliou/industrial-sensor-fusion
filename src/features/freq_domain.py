import numpy as np
from scipy import stats

_SCALAR_FEATURES = [
    "dominant_freq", "spectral_centroid", "spectral_entropy",
    "band_energy_low", "band_energy_mid", "band_energy_high",
]


def _feature_names(top_k: int) -> list[str]:
    return [f"fft_bin_{i:02d}" for i in range(top_k)] + _SCALAR_FEATURES


def compute_freq_features(
    signal: np.ndarray,
    fs: float,
    top_k: int = 20,
) -> dict[str, float]:
    # top_k is the number of fixed-width frequency bands (kept this name for config
    # compatibility). Each fft_bin_XX is the mean magnitude in a fixed band spanning
    # [0, Nyquist], so a given column means the same frequency range in every cycle —
    # unlike selecting the per-cycle top peaks, where a column's frequency drifted row to row.
    n = len(signal)
    if n == 0:
        # Build the empty dict from top_k so the column set matches the normal path even if
        # freq_domain_top_k is changed in config.
        return dict.fromkeys(_feature_names(top_k), 0.0)

    fft_mag = np.abs(np.fft.rfft(signal)) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    nyquist = fs / 2.0

    edges = np.linspace(0.0, nyquist, top_k + 1)
    features: dict[str, float] = {}
    for i in range(top_k):
        lo, hi = edges[i], edges[i + 1]
        mask = (freqs >= lo) & (freqs <= hi) if i == top_k - 1 else (freqs >= lo) & (freqs < hi)
        features[f"fft_bin_{i:02d}"] = float(fft_mag[mask].mean()) if mask.any() else 0.0

    features["dominant_freq"] = float(freqs[np.argmax(fft_mag)])

    mag_sum = fft_mag.sum()
    features["spectral_centroid"] = (
        float(np.sum(freqs * fft_mag) / mag_sum) if mag_sum > 0 else 0.0
    )

    power = fft_mag ** 2
    total_power = power.sum()
    if total_power > 0:
        features["spectral_entropy"] = float(stats.entropy(power / total_power + 1e-12))
    else:
        # A dead/flat signal has no spectral content. Without this guard the uniform
        # fallback would report MAX entropy — the opposite of reality for a failed sensor.
        features["spectral_entropy"] = 0.0

    band_limits = [(0, 10), (10, 30), (30, 50)]
    band_names = ["band_energy_low", "band_energy_mid", "band_energy_high"]
    for (lo, hi), name in zip(band_limits, band_names):
        mask = (freqs >= lo) & (freqs < hi)
        features[name] = float(np.sum(fft_mag[mask] ** 2))

    return features
