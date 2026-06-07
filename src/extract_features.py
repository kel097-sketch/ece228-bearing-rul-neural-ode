import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis, skew
from tqdm import tqdm

from config import FS
from utils import ensure_dir, safe_read_vibration_csv

EPS = 1e-12
WAVELET_NAME = "db4"
WAVELET_LEVEL = 3
WAVELET_MAX_POINTS = 8192


def resolve_csv_path(file_path: str, metadata_path: Path) -> Path:
    path = Path(file_path)
    if path.exists():
        return path
    root_relative = Path.cwd() / path
    if root_relative.exists():
        return root_relative
    metadata_relative = metadata_path.parent / path
    if metadata_relative.exists():
        return metadata_relative
    return path


def time_domain_features(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x**2)))
    abs_mean = float(np.mean(np.abs(x)))
    max_abs = float(np.max(np.abs(x)))
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "rms": rms,
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "peak_to_peak": float(np.ptp(x)),
        "abs_mean": abs_mean,
        "skewness": float(skew(x, bias=False)) if len(x) > 2 else 0.0,
        "kurtosis": float(kurtosis(x, fisher=True, bias=False)) if len(x) > 3 else 0.0,
        "energy": float(np.sum(x**2)),
        "max_abs": max_abs,
        "crest_factor": float(max_abs / (rms + EPS)),
        "impulse_factor": float(max_abs / (abs_mean + EPS)),
        "shape_factor": float(rms / (abs_mean + EPS)),
    }


def frequency_domain_features(x: np.ndarray, fs: float = FS) -> dict:
    x = np.asarray(x, dtype=np.float64)
    fft_values = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    power = np.abs(fft_values) ** 2
    spectral_energy = float(np.sum(power))

    if spectral_energy <= EPS:
        probabilities = np.ones_like(power) / max(len(power), 1)
        spectral_centroid = 0.0
        dominant_frequency = 0.0
    else:
        probabilities = power / (spectral_energy + EPS)
        spectral_centroid = float(np.sum(freqs * power) / (spectral_energy + EPS))
        dominant_frequency = float(freqs[int(np.argmax(power))])

    spectral_entropy = float(-np.sum(probabilities * np.log2(probabilities + EPS)))

    bands = {
        "0_2k": (0.0, 2000.0),
        "2k_5k": (2000.0, 5000.0),
        "5k_10k": (5000.0, 10000.0),
        "10k_nyquist": (10000.0, fs / 2.0 + EPS),
    }
    features = {
        "spectral_energy": spectral_energy,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "dominant_frequency": dominant_frequency,
    }
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_energy = float(np.sum(power[mask]))
        features[f"band_energy_{name}"] = band_energy
    for name in bands:
        features[f"band_ratio_{name}"] = float(features[f"band_energy_{name}"] / (spectral_energy + EPS))
    return features


def wavelet_packet_features(
    x: np.ndarray,
    wavelet: str = WAVELET_NAME,
    level: int = WAVELET_LEVEL,
) -> dict:
    x = np.asarray(x, dtype=np.float64)
    packet = pywt.WaveletPacket(data=x, wavelet=wavelet, mode="symmetric", maxlevel=level)
    nodes = packet.get_level(level, order="freq")
    features = {}
    total_energy = 0.0
    energies = []
    for i, node in enumerate(nodes, start=1):
        coeff = np.asarray(node.data, dtype=np.float64)
        energy = float(np.sum(coeff**2))
        rms = float(np.sqrt(np.mean(coeff**2))) if coeff.size else 0.0
        energies.append(energy)
        total_energy += energy
        features[f"wpt_l{level}_b{i}_energy"] = energy
        features[f"wpt_l{level}_b{i}_rms"] = rms
    for i, energy in enumerate(energies, start=1):
        features[f"wpt_l{level}_b{i}_energy_ratio"] = float(energy / (total_energy + EPS))
    return features


def wavelet_coefficient_features(
    x: np.ndarray,
    wavelet: str = WAVELET_NAME,
    level: int = WAVELET_LEVEL,
) -> dict:
    x = np.asarray(x, dtype=np.float64)
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    use_level = min(level, max_level)
    coeffs = pywt.wavedec(x, wavelet=wavelet, mode="symmetric", level=use_level)
    labels = [f"ca{use_level}", *[f"cd{i}" for i in range(use_level, 0, -1)]]
    features = {}
    for label, coeff in zip(labels, coeffs):
        coeff = np.asarray(coeff, dtype=np.float64)
        if coeff.size == 0:
            features[f"wav_{label}_mean"] = 0.0
            features[f"wav_{label}_var"] = 0.0
            features[f"wav_{label}_energy"] = 0.0
            continue
        features[f"wav_{label}_mean"] = float(np.mean(coeff))
        features[f"wav_{label}_var"] = float(np.var(coeff))
        features[f"wav_{label}_energy"] = float(np.sum(coeff**2))
    return features


def wavelet_features(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if len(x) > WAVELET_MAX_POINTS:
        step = int(np.ceil(len(x) / WAVELET_MAX_POINTS))
        x = x[::step]
    features = {}
    features.update(wavelet_packet_features(x))
    features.update(wavelet_coefficient_features(x))
    return features


def channel_features(x: np.ndarray, prefix: str, include_wavelet: bool = False) -> dict:
    features = {}
    for name, value in time_domain_features(x).items():
        features[f"{prefix}_{name}"] = value
    for name, value in frequency_domain_features(x).items():
        features[f"{prefix}_{name}"] = value
    if include_wavelet:
        for name, value in wavelet_features(x).items():
            features[f"{prefix}_{name}"] = value
    return features


def extract_features(metadata: str, out: str, include_wavelet: bool = False) -> pd.DataFrame:
    metadata_path = Path(metadata)
    meta_df = pd.read_csv(metadata_path)
    rows = []
    failed = []

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Extracting features"):
        row_dict = row.to_dict()
        csv_path = resolve_csv_path(str(row_dict["file_path"]), metadata_path)
        try:
            horizontal, vertical = safe_read_vibration_csv(csv_path)
            features = {}
            features.update(channel_features(horizontal, "h", include_wavelet=include_wavelet))
            features.update(channel_features(vertical, "v", include_wavelet=include_wavelet))
            row_dict.update(features)
            rows.append(row_dict)
        except Exception as exc:
            failed.append((str(csv_path), str(exc)))

    if not rows:
        raise RuntimeError("No feature rows were successfully extracted.")

    features_df = pd.DataFrame(rows)
    out_path = Path(out)
    ensure_dir(out_path.parent)
    features_df.to_csv(out_path, index=False)

    print(f"Saved features to {out_path}")
    print(f"Number of successfully processed files: {len(rows)}")
    print(f"Number of failed files: {len(failed)}")
    if failed:
        print("Failed files:")
        for path, error in failed:
            print(f"  {path}: {error}")

    return features_df


def parse_args():
    parser = argparse.ArgumentParser(description="Extract vibration-derived features.")
    parser.add_argument("--metadata", default="processed/metadata.csv")
    parser.add_argument("--out", default="processed/features.csv")
    parser.add_argument(
        "--include_wavelet",
        action="store_true",
        help="Also extract db4 level-3 WPT/DWT wavelet features. Without this flag, only the 52 original time/frequency features are written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extract_features(args.metadata, args.out, include_wavelet=args.include_wavelet)


if __name__ == "__main__":
    main()
