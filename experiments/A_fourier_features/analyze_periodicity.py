#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate dominant temporal periodicities from spatiotemporal fields via:
1) Autocorrelation function (ACF) peaks
2) Power spectral density (periodogram) peaks

Saves results to: experiments/A_no_time/runs/data_feature
"""

import os
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import scipy.io
import h5py
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple


# -------------------------
# MatReader (as provided)
# -------------------------
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except Exception:
            self.data = h5py.File(self.file_path, 'r')
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field_name):
        x = self.data[field_name]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# -------------------------
# Helpers
# -------------------------
def ensure_time_hw(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure x is shaped (T, H, W).
    Supports possible shapes: (T,H,W), (T,H,W,1), (H,W,T), etc.
    We try to guess based on which dim is largest (assume T=~2920).
    """
    x = x.detach().cpu()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]  # (T,H,W)

    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor (T,H,W) or (T,H,W,1). Got shape {tuple(x.shape)}")

    # Heuristic: time dimension is the one close to 2920 or generally the largest
    dims = list(x.shape)
    t_dim = int(np.argmax(dims))
    if t_dim != 0:
        x = x.permute(t_dim, *[d for d in range(3) if d != t_dim])  # move time to front

    return x  # (T,H,W)


def detrend_series(s: np.ndarray, do_detrend: bool = True) -> np.ndarray:
    """
    Remove mean and (optionally) linear trend.
    """
    s = s.astype(np.float64)
    s = s - np.mean(s)
    if do_detrend:
        s = signal.detrend(s, type="linear")
    return s


def acf_fft(s: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Fast autocorrelation via FFT. Returns ACF[0..max_lag] normalized so ACF[0]=1.
    """
    n = len(s)
    # next pow2 for speed
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(s, n=nfft)
    acf = np.fft.irfft(f * np.conj(f), n=nfft)[:n]
    # unbiased/biased normalization: use biased for stability
    acf = acf / (acf[0] + 1e-12)
    return acf[: max_lag + 1]


def find_top_acf_peak(acf: np.ndarray, min_lag: int = 1) -> tuple[int, float]:
    """
    Find the dominant ACF peak lag (excluding lag 0).
    Returns (lag, value).
    """
    if len(acf) <= min_lag:
        return (-1, float("nan"))
    # exclude lag 0 and very small lags if needed
    idx = np.argmax(acf[min_lag:]) + min_lag
    return int(idx), float(acf[idx])


def periodogram_peak(
    s: np.ndarray,
    fs: float,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> Tuple[float, float]:
    """
    Compute periodogram and return dominant frequency peak (Hz in units of cycles/day if fs is samples/day).
    Returns (f_peak, pxx_peak).
    """
    f, pxx = signal.periodogram(s, fs=fs, window="hann", detrend=False, scaling="density")
    if fmax is None:
        fmax = f.max()
    mask = (f >= fmin) & (f <= fmax)
    f2, p2 = f[mask], pxx[mask]
    if len(f2) == 0:
        return (float("nan"), float("nan"))
    i = int(np.argmax(p2))
    return float(f2[i]), float(p2[i])


# -------------------------
# Main analysis
# -------------------------
def main():
    # ====== User config (edit if needed) ======
    # Set these to your repo paths:
    PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    SCRIPT_DIR = Path(__file__).resolve().parent

    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_X_PATH = DATA_DIR / "interp_train_x_SSP_TLshape_ndrz10.mat"
    TEST_X_PATH  = DATA_DIR / "interp_test_x_SSP_TLshape_ndrz10.mat"

    # Output folder (as requested)
    OUT_DIR = PROJECT_ROOT / "experiments" / "A_fourier_features" / "data_feature"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Temporal sampling settings
    SAMPLES_PER_DAY = 8.0           # every 3 hours
    FS = SAMPLES_PER_DAY            # samples/day for periodogram frequency unit: cycles/day

    # Compute settings
    MAX_LAG_DAYS = 60               # analyze ACF up to 60 days
    MAX_LAG = int(MAX_LAG_DAYS * SAMPLES_PER_DAY)  # in samples
    MIN_LAG = 1                     # exclude lag 0
    DO_DETREND = True               # recommended
    N_POINTS = 500                  # random spatial points to analyze (increase if feasible)
    SEED = 2025

    # Optional frequency search range for periodogram (cycles/day)
    # If you only care daily-ish to seasonal-ish, set fmin small but nonzero.
    FMIN = 0.0
    FMAX = FS / 2.0                 # Nyquist (cycles/day) = 4.0 when FS=8

    # ====== Load data ======
    print("[INFO] Loading data...")
    x_train_part = MatReader(str(TRAIN_X_PATH)).read_field("train_x")
    x_test_part  = MatReader(str(TEST_X_PATH)).read_field("test_x")
    x_all = torch.cat([x_train_part, x_test_part], dim=0)

    # Ensure (T,H,W)
    x_all = ensure_time_hw(x_all)
    T, H, W = x_all.shape
    print(f"[INFO] x_all shape = (T,H,W)=({T},{H},{W})")

    # ====== Choose spatial points ======
    rng = np.random.default_rng(SEED)
    total_points = H * W
    n = min(N_POINTS, total_points)
    flat_indices = rng.choice(total_points, size=n, replace=False)
    points = [(int(idx // W), int(idx % W)) for idx in flat_indices]
    print(f"[INFO] Sampling {n} spatial points out of {total_points}")

    # ====== Per-point analysis ======
    rows = []
    # For averaged periodogram plot
    all_pxx = []
    f_ref = None

    for (i, j) in points:
        s = x_all[:, i, j].numpy()
        s = detrend_series(s, do_detrend=DO_DETREND)

        # ACF
        acf = acf_fft(s, max_lag=MAX_LAG)
        lag_peak, acf_peak_val = find_top_acf_peak(acf, min_lag=MIN_LAG)
        period_samples_from_acf = lag_peak if lag_peak >= 0 else np.nan
        period_days_from_acf = (period_samples_from_acf / SAMPLES_PER_DAY) if lag_peak >= 0 else np.nan

        # Periodogram
        f, pxx = signal.periodogram(s, fs=FS, window="hann", detrend=False, scaling="density")
        if f_ref is None:
            f_ref = f
        # Restrict frequency range for peak finding
        mask = (f >= FMIN) & (f <= FMAX)
        f2, p2 = f[mask], pxx[mask]
        if len(f2) > 0:
            k = int(np.argmax(p2))
            f_peak = float(f2[k])
            p_peak = float(p2[k])
            period_days_from_psd = (1.0 / f_peak) if f_peak > 1e-12 else np.inf
        else:
            f_peak, p_peak, period_days_from_psd = np.nan, np.nan, np.nan

        all_pxx.append(pxx)

        rows.append({
            "i": i, "j": j,
            "acf_peak_lag_samples": period_samples_from_acf,
            "acf_peak_period_days": period_days_from_acf,
            "acf_peak_value": acf_peak_val,
            "psd_peak_freq_cyc_per_day": f_peak,
            "psd_peak_period_days": period_days_from_psd,
            "psd_peak_power": p_peak,
        })

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "per_point_peaks.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] {csv_path}")

    # ====== Aggregate stats ======
    # ACF periods (days)
    acf_periods = df["acf_peak_period_days"].replace([np.inf, -np.inf], np.nan).dropna().values
    psd_periods = df["psd_peak_period_days"].replace([np.inf, -np.inf], np.nan).dropna().values
    psd_freqs   = df["psd_peak_freq_cyc_per_day"].replace([np.inf, -np.inf], np.nan).dropna().values

    def summarize(arr, name):
        if len(arr) == 0:
            return {f"{name}_count": 0}
        return {
            f"{name}_count": int(len(arr)),
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_median": float(np.median(arr)),
            f"{name}_p10": float(np.quantile(arr, 0.10)),
            f"{name}_p90": float(np.quantile(arr, 0.90)),
        }

    summary = {
        "T": int(T), "H": int(H), "W": int(W),
        "samples_per_day": SAMPLES_PER_DAY,
        "max_lag_days": MAX_LAG_DAYS,
        "n_points": int(n),
        "detrend": bool(DO_DETREND),
        "acf_period_days": summarize(acf_periods, "acf_period_days"),
        "psd_period_days": summarize(psd_periods, "psd_period_days"),
        "psd_freq_cyc_per_day": summarize(psd_freqs, "psd_freq_cyc_per_day"),
    }
    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[SAVE] {json_path}")

    # ====== Plots ======
    # 1) Histogram of ACF peak lags (in days)
    plt.figure()
    if len(acf_periods) > 0:
        plt.hist(acf_periods, bins=40)
        plt.xlabel("Dominant period from ACF (days)")
        plt.ylabel("Count")
        plt.title("ACF dominant period distribution")
    else:
        plt.text(0.1, 0.5, "No valid ACF periods", transform=plt.gca().transAxes)
        plt.axis("off")
    p1 = OUT_DIR / "period_from_acf_hist.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"[SAVE] {p1}")

    # 2) Histogram of PSD peak periods (in days)
    plt.figure()
    if len(psd_periods) > 0 and np.isfinite(psd_periods).any():
        finite = psd_periods[np.isfinite(psd_periods)]
        plt.hist(finite, bins=40)
        plt.xlabel("Dominant period from periodogram (days)")
        plt.ylabel("Count")
        plt.title("Periodogram dominant period distribution")
    else:
        plt.text(0.1, 0.5, "No valid PSD periods", transform=plt.gca().transAxes)
        plt.axis("off")
    p2 = OUT_DIR / "period_from_periodogram_hist.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"[SAVE] {p2}")

    # 3) Mean periodogram (average across points)
    if f_ref is not None and len(all_pxx) > 0:
        P = np.stack(all_pxx, axis=0)
        mean_pxx = np.mean(P, axis=0)

        plt.figure()
        plt.plot(f_ref, mean_pxx)
        plt.xlabel("Frequency (cycles/day)")
        plt.ylabel("PSD (mean)")
        plt.title("Mean periodogram across sampled spatial points")
        plt.xlim([FMIN, FMAX])
        p3 = OUT_DIR / "periodogram_mean.png"
        plt.tight_layout()
        plt.savefig(p3, dpi=200)
        plt.close()
        print(f"[SAVE] {p3}")

    print("[DONE] Analysis complete.")
    print(f"[INFO] Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
