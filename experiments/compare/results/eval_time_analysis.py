# -*- coding: utf-8 -*-
"""
eval_time_analysis.py (3-model version)

Models:
  - Baseline A: SSP only (Cin=1)
  - A' (time scalar): SSP + t_scalar (Cin=2)
  - A'' (time PE): SSP + [sin/cos day, sin/cos year] (Cin=5)

Outputs:
  dist/      : hist + CDF for RelL2/RMSE/MAE + summary table txt
  time_curve/: error vs global time index (raw + moving average)
  season/    : seasonal + monthly grouped stats + win-rate
  maps/      : mean |error| maps + diff maps vs baseline
  metrics_and_preds.npz

Assumptions:
  - test split corresponds to global indices [ntrain .. ntrain+ntest-1]
  - normalizers saved from TRAIN ONLY in a .pt dict with keys:
      x_mean, x_std, y_mean, y_std, eps
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from timeit import default_timer

from pathlib import Path
import sys

# --------------------------------------------------
# Robust project root discovery (find folder with src/)
# --------------------------------------------------
THIS_FILE = Path(__file__).resolve()
p = THIS_FILE
PROJECT_ROOT = None

for _ in range(6):  # 最多向上找 6 层
    if (p / "src").exists():
        PROJECT_ROOT = p
        break
    p = p.parent

if PROJECT_ROOT is None:
    raise RuntimeError("Cannot find project root containing 'src/'")

sys.path.insert(0, str(PROJECT_ROOT))
print(f"[eval] PROJECT_ROOT = {PROJECT_ROOT}")

from src.io_mat import MatReader
from src.normalizer import UnitGaussianNormalizer
from src.models.fno2d import FNO2d


# -----------------------------
# metrics
# -----------------------------
def rel_l2(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    num = np.linalg.norm((pred - gt).ravel(), ord=2)
    den = np.linalg.norm(gt.ravel(), ord=2) + eps
    return float(num / den)

def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))

def summarize(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "p50": np.nan, "p90": np.nan, "p99": np.nan, "max": np.nan}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


# -----------------------------
# time features
# -----------------------------
def build_time_scalar(global_idx: torch.Tensor, nt_total: int) -> torch.Tensor:
    # [0,1]
    return (global_idx / (nt_total - 1.0)).view(-1, 1)  # (N,1)

def build_time_pe(global_idx: torch.Tensor, nt_total: int, samples_per_day: int = 8) -> torch.Tensor:
    # (N,4): sin/cos(day), sin/cos(year)
    day_phase = (global_idx.remainder(float(samples_per_day))) / float(samples_per_day)
    year_phase = global_idx / (nt_total - 1.0)
    two_pi = 2.0 * np.pi
    return torch.stack([
        torch.sin(two_pi * day_phase),
        torch.cos(two_pi * day_phase),
        torch.sin(two_pi * year_phase),
        torch.cos(two_pi * year_phase),
    ], dim=-1)


# -----------------------------
# month/season grouping (non-leap 365)
# -----------------------------
def month_from_global_idx(global_idx: np.ndarray, samples_per_day: int = 8) -> np.ndarray:
    month_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=int)
    cum = np.cumsum(month_days)
    day = (global_idx // samples_per_day).astype(int)  # 0..364
    month = np.searchsorted(cum, day + 1) + 1          # 1..12
    return month

def season_from_month(month: np.ndarray) -> np.ndarray:
    season = np.empty_like(month, dtype=object)
    season[np.isin(month, [12, 1, 2])] = "DJF"
    season[np.isin(month, [3, 4, 5])]  = "MAM"
    season[np.isin(month, [6, 7, 8])]  = "JJA"
    season[np.isin(month, [9, 10, 11])] = "SON"
    return season


# -----------------------------
# normalizers
# -----------------------------
def load_normalizers(norm_path: Path, device: torch.device):
    norm = torch.load(str(norm_path), map_location="cpu")

    x_norm = UnitGaussianNormalizer(torch.zeros(1))
    y_norm = UnitGaussianNormalizer(torch.zeros(1))

    x_norm.mean = norm["x_mean"]
    x_norm.std  = norm["x_std"]
    y_norm.mean = norm["y_mean"]
    y_norm.std  = norm["y_std"]

    eps = float(norm.get("eps", 1e-5))
    x_norm.eps = eps
    y_norm.eps = eps

    x_norm.mean = x_norm.mean.to(device)
    x_norm.std  = x_norm.std.to(device)
    y_norm.mean = y_norm.mean.to(device)
    y_norm.std  = y_norm.std.to(device)

    return x_norm, y_norm


# -----------------------------
# inference
# -----------------------------
@torch.no_grad()
def predict_model(model, x_in: torch.Tensor, y_norm: UnitGaussianNormalizer, device: torch.device, batch_size: int):
    model.eval()
    preds = []
    N = x_in.shape[0]
    for i in range(0, N, batch_size):
        xb = x_in[i:i+batch_size].to(device)
        out = model(xb)
        # allow (B,H,W,1) or (B,H,W)
        if out.dim() == 4 and out.size(-1) == 1:
            out = out.squeeze(-1)
        out_phys = y_norm.decode(out)
        preds.append(out_phys.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)  # (N,H,W)


# -----------------------------
# plotting
# -----------------------------
def save_hist_and_cdf(values: np.ndarray, title: str, out_png: Path, bins: int = 60):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.hist(v, bins=bins)
    plt.title(f"{title} - Hist")
    plt.xlabel("Error")
    plt.ylabel("Count")

    plt.subplot(1,2,2)
    xs = np.sort(v)
    ys = np.linspace(0, 1, len(xs), endpoint=True)
    plt.plot(xs, ys)
    plt.title(f"{title} - CDF")
    plt.xlabel("Error")
    plt.ylabel("CDF")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def moving_average(y: np.ndarray, k: int):
    if k <= 1:
        return y
    ypad = np.pad(y, (k//2, k-1-k//2), mode="edge")
    return np.convolve(ypad, np.ones(k)/k, mode="valid")

def save_time_curve(t: np.ndarray, curves: dict, title: str, out_png: Path, smooth_win: int = 25):
    plt.figure(figsize=(12,4))
    for name, y in curves.items():
        plt.plot(t, y, label=name, alpha=0.35)
        ys = moving_average(y, smooth_win) if smooth_win else y
        plt.plot(t, ys, label=f"{name} MA{smooth_win}", linewidth=2)
    plt.title(title)
    plt.xlabel("Global time index (absolute)")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_error_map(arr2d: np.ndarray, title: str, out_png: Path):
    plt.figure(figsize=(6,4))
    im = plt.imshow(arr2d, aspect="auto")
    plt.title(title)
    plt.colorbar(im, shrink=0.85)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# win-rate
# -----------------------------
def win_rate(a: np.ndarray, b: np.ndarray) -> float:
    # fraction where a < b
    return float(np.mean(a < b))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_x", type=str, default="data/interp_test_x_SSP_TLshape_ndrz10.mat")
    ap.add_argument("--test_y", type=str, default="data/test_y_SSP_TLshape_ndrz10.mat")
    ap.add_argument("--norm",   type=str, default="normalizers/ssp_tl_norm_train2336_ndrz10.pt")
    ap.add_argument("--out",    type=str, default="results/time_analysis/compare3")

    # model paths (optional but at least one required)
    ap.add_argument("--model_baseline", type=str, default="")
    ap.add_argument("--model_scalar",   type=str, default="")
    ap.add_argument("--model_pe",       type=str, default="")

    # dataset/time config
    ap.add_argument("--ntrain", type=int, default=2336)
    ap.add_argument("--ntest",  type=int, default=584)
    ap.add_argument("--nt_total", type=int, default=2920)
    ap.add_argument("--samples_per_day", type=int, default=8)

    # model hyperparams (must match training)
    ap.add_argument("--modes1", type=int, default=32)
    ap.add_argument("--modes2", type=int, default=128)
    ap.add_argument("--width",  type=int, default=64)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dirs
    dist_dir = out_dir / "dist"; dist_dir.mkdir(parents=True, exist_ok=True)
    time_dir = out_dir / "time_curve"; time_dir.mkdir(parents=True, exist_ok=True)
    season_dir = out_dir / "season"; season_dir.mkdir(parents=True, exist_ok=True)
    map_dir = out_dir / "maps"; map_dir.mkdir(parents=True, exist_ok=True)

    # load test
    print("Loading test data...")
    x_test = MatReader(args.test_x).read_field("test_x")[:args.ntest]   # (N,H,W)
    y_test = MatReader(args.test_y).read_field("test_y")[:args.ntest]   # (N,H,W)
    x_test = x_test.unsqueeze(-1)                                       # (N,H,W,1)

    N, H, W, _ = x_test.shape
    y_test_np = y_test.cpu().numpy()

    # global idx for test
    idx_test = torch.arange(args.ntrain, args.ntrain + args.ntest, dtype=torch.float32)
    idx_test_np = idx_test.numpy().astype(int)

    # month & season
    month = month_from_global_idx(idx_test_np, args.samples_per_day)    # 1..12
    season = season_from_month(month)                                   # DJF/MAM/JJA/SON

    # load normalizers + normalize SSP
    x_norm, y_norm = load_normalizers(Path(args.norm), device=device)
    x_ssp = x_test[..., 0:1].to(device)                                 # (N,H,W,1)
    x_ssp = x_norm.encode(x_ssp)

    # build inputs
    x_inputs = {}
    if args.model_baseline:
        x_inputs["baseline"] = x_ssp                                    # (N,H,W,1)

    if args.model_scalar:
        t_scalar = build_time_scalar(idx_test.to(device), args.nt_total).view(N,1,1,1).repeat(1,H,W,1)
        x_inputs["time_scalar"] = torch.cat([x_ssp, t_scalar], dim=-1)  # (N,H,W,2)

    if args.model_pe:
        t_pe = build_time_pe(idx_test.to(device), args.nt_total, args.samples_per_day).view(N,1,1,4).repeat(1,H,W,1)
        x_inputs["time_pe"] = torch.cat([x_ssp, t_pe], dim=-1)          # (N,H,W,5)

    if not x_inputs:
        raise RuntimeError("No model paths provided. Use --model_baseline / --model_scalar / --model_pe.")

    # load + predict
    preds = {}
    print("Running inference...")
    t0 = default_timer()

    if args.model_baseline:
        m = FNO2d(args.modes1, args.modes2, args.width, in_dim=1).to(device)
        m.load_state_dict(torch.load(args.model_baseline, map_location=device))
        preds["A_baseline"] = predict_model(m, x_inputs["baseline"], y_norm, device, args.batch_size)

    if args.model_scalar:
        m = FNO2d(args.modes1, args.modes2, args.width, in_dim=2).to(device)
        m.load_state_dict(torch.load(args.model_scalar, map_location=device))
        preds["Aprime_time_scalar"] = predict_model(m, x_inputs["time_scalar"], y_norm, device, args.batch_size)

    if args.model_pe:
        m = FNO2d(args.modes1, args.modes2, args.width, in_dim=5).to(device)
        m.load_state_dict(torch.load(args.model_pe, map_location=device))
        preds["Adblprime_time_pe"] = predict_model(m, x_inputs["time_pe"], y_norm, device, args.batch_size)

    print(f"Inference done in {default_timer()-t0:.2f}s")

    # per-sample metrics
    metrics = {}
    for name, p in preds.items():
        rels, rmses, maes = [], [], []
        for i in range(N):
            gt = y_test_np[i]
            pr = p[i]
            rels.append(rel_l2(pr, gt))
            rmses.append(rmse(pr, gt))
            maes.append(mae(pr, gt))
        metrics[name] = {
            "rel_l2": np.array(rels),
            "rmse":   np.array(rmses),
            "mae":    np.array(maes),
        }

    # summary txt
    summary_path = dist_dir / "summary_metrics.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Overall metrics on TEST (per-sample)\n")
        f.write(f"ntest={N}, global_idx=[{args.ntrain}..{args.ntrain+args.ntest-1}]\n\n")
        for name in preds.keys():
            s1 = summarize(metrics[name]["rel_l2"])
            s2 = summarize(metrics[name]["rmse"])
            s3 = summarize(metrics[name]["mae"])
            f.write(f"== {name} ==\n")
            f.write(f"RelL2: mean={s1['mean']:.6f}, p50={s1['p50']:.6f}, p90={s1['p90']:.6f}, p99={s1['p99']:.6f}, max={s1['max']:.6f}\n")
            f.write(f"RMSE : mean={s2['mean']:.6f}, p50={s2['p50']:.6f}, p90={s2['p90']:.6f}, p99={s2['p99']:.6f}, max={s2['max']:.6f}\n")
            f.write(f"MAE  : mean={s3['mean']:.6f}, p50={s3['p50']:.6f}, p90={s3['p90']:.6f}, p99={s3['p99']:.6f}, max={s3['max']:.6f}\n\n")
    print("Saved:", summary_path)

    # dist plots
    for name in preds.keys():
        save_hist_and_cdf(metrics[name]["rel_l2"], f"{name} RelL2", dist_dir / f"{name}_relL2_hist_cdf.png")
        save_hist_and_cdf(metrics[name]["rmse"],   f"{name} RMSE(dB)", dist_dir / f"{name}_rmse_hist_cdf.png")
        save_hist_and_cdf(metrics[name]["mae"],    f"{name} MAE(dB)",  dist_dir / f"{name}_mae_hist_cdf.png")

    # time curves (use global idx as x-axis)
    t_axis = idx_test_np
    save_time_curve(
        t_axis,
        {k: metrics[k]["rel_l2"] for k in preds.keys()},
        "RelL2 vs Global Time Index",
        time_dir / "relL2_vs_time.png",
        smooth_win=25
    )
    save_time_curve(
        t_axis,
        {k: metrics[k]["rmse"] for k in preds.keys()},
        "RMSE(dB) vs Global Time Index",
        time_dir / "rmse_vs_time.png",
        smooth_win=25
    )

    # seasonal + monthly stats + win-rate
    def grouped_report(group_name: str, group_ids: list, group_values: np.ndarray, group_labels: list[str], out_txt: Path):
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"{group_name} grouped stats (metric: RMSE)\n\n")
            for g, lab in zip(group_ids, group_labels):
                idxs = np.where(group_values == g)[0]
                f.write(f"== {lab} == n={len(idxs)}\n")
                if len(idxs) == 0:
                    f.write("  (no samples)\n\n")
                    continue
                for name in preds.keys():
                    arr = metrics[name]["rmse"][idxs]
                    ss = summarize(arr)
                    f.write(f"  {name}: mean={ss['mean']:.4f}, p50={ss['p50']:.4f}, p90={ss['p90']:.4f}, p99={ss['p99']:.4f}\n")

                # win-rate vs baseline if baseline exists
                if "A_baseline" in preds:
                    base = metrics["A_baseline"]["rel_l2"][idxs]
                    for name in preds.keys():
                        if name == "A_baseline":
                            continue
                        wr = win_rate(metrics[name]["rel_l2"][idxs], base)
                        f.write(f"  WinRate({name} < baseline) on RelL2: {wr*100:.2f}%\n")
                f.write("\n")

    # seasons
    season_ids = ["DJF","MAM","JJA","SON"]
    grouped_report(
        "Season",
        season_ids,
        season,
        season_ids,
        season_dir / "seasonal_stats_rmse_and_winrate.txt"
    )

    # months 1..12
    month_ids = list(range(1,13))
    month_labels = [f"M{m:02d}" for m in month_ids]
    grouped_report(
        "Month",
        month_ids,
        month,
        month_labels,
        season_dir / "monthly_stats_rmse_and_winrate.txt"
    )

    # maps: mean abs error + diff vs baseline
    abs_maps = {}
    for name, p in preds.items():
        abs_err = np.abs(p - y_test_np)          # (N,H,W)
        abs_maps[name] = abs_err.mean(axis=0)    # (H,W)
        save_error_map(abs_maps[name], f"{name} Mean |Error|", map_dir / f"{name}_mean_abs_err.png")

    if "A_baseline" in abs_maps:
        base_map = abs_maps["A_baseline"]
        for name in abs_maps.keys():
            if name == "A_baseline":
                continue
            diff = abs_maps[name] - base_map
            save_error_map(diff, f"Mean|Err| Diff: {name} - baseline", map_dir / f"diff_{name}_minus_baseline.png")

    # save npz for later custom analysis
    npz_path = out_dir / "metrics_and_preds.npz"
    save_dict = {
        "test_global_idx": idx_test_np,
        "month": month.astype(int),
        "season": season.astype("U3"),
        "y_test": y_test_np,
    }
    for name in preds.keys():
        save_dict[f"{name}_pred"] = preds[name]
        save_dict[f"{name}_relL2"] = metrics[name]["rel_l2"]
        save_dict[f"{name}_rmse"] = metrics[name]["rmse"]
        save_dict[f"{name}_mae"] = metrics[name]["mae"]
    np.savez_compressed(npz_path, **save_dict)

    print("Saved all outputs to:", out_dir)
    print("NPZ:", npz_path)


if __name__ == "__main__":
    main()
