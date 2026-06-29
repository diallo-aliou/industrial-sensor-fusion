import numpy as np
from scipy import stats

_FEATURE_NAMES = [
    "mean", "std", "min", "max", "peak_to_peak",
    "rms", "crest_factor", "skewness", "kurtosis",
]


def compute_time_features(signal: np.ndarray) -> dict[str, float]:
    if len(signal) == 0:
        return dict.fromkeys(_FEATURE_NAMES, 0.0)

    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    crest_factor = peak / rms if rms > 0 else 0.0

    return {
        "mean":         float(np.mean(signal)),
        "std":          float(np.std(signal)),
        "min":          float(np.min(signal)),
        "max":          float(np.max(signal)),
        "peak_to_peak": float(np.max(signal) - np.min(signal)),
        "rms":          rms,
        "crest_factor": crest_factor,
        "skewness":     float(np.nan_to_num(stats.skew(signal),     nan=0.0)),
        "kurtosis":     float(np.nan_to_num(stats.kurtosis(signal), nan=0.0)),
    }


def compute_trend_features(signal: np.ndarray, fs: float) -> dict[str, float]:
    # Slope (rate of change) and integral (accumulated area) over the cycle, in physical
    # time. Meaningful for slow thermodynamic channels (temperature rise rate, total heat)
    # where a single linear trend and area describe the signal better than spikiness stats.
    if len(signal) < 2:
        return {"slope": 0.0, "integral": 0.0}
    dt = 1.0 / fs
    t = np.arange(len(signal)) * dt
    slope = float(np.polyfit(t, signal, 1)[0])
    integral = float((signal.sum() - 0.5 * (signal[0] + signal[-1])) * dt)  # trapezoidal
    return {"slope": slope, "integral": integral}
