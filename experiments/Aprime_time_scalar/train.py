"""
Training a Fourier Neural Operator (FNO) for predicting underwater acoustic transmission loss fields
from sound speed profiles (SSP) with scalar time conditioning.

Model: A' (Aprime_time_scalar)
Input channels:
  - SSP (1 channel, normalized)
  - Time scalar (1 channel, raw in [0,1], global continuity)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.io_mat import MatReader
from src.normalizer import UnitGaussianNormalizer
from src.models.fno2d import FNO2d, LpLoss, H1Loss

torch.manual_seed(0)
np.random.seed(0)

# =============================================================================
# Configuration
# =============================================================================
ntrain = 2336
ntest = 584
nt_total = 2920  # full year length (train+test global timeline)

batch_size = 20
learning_rate = 0.001
epochs = 100
step_size = 50
gamma = 0.5

modes1 = 32
modes2 = 128
width = 64

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = SCRIPT_DIR / "runs" / "shuffled" / "modes1_32_modes2_128_epoch_100"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_X_PATH = DATA_DIR / "interp_train_x_SSP_TLshape_ndrz10.mat"
TRAIN_Y_PATH = DATA_DIR / "train_y_SSP_TLshape_ndrz10.mat"
TEST_X_PATH  = DATA_DIR / "interp_test_x_SSP_TLshape_ndrz10.mat"
TEST_Y_PATH  = DATA_DIR / "test_y_SSP_TLshape_ndrz10.mat"

# Shared normalizers (train-split statistics only)
NORM_DIR = PROJECT_ROOT / "normalizers"
NORM_DIR.mkdir(parents=True, exist_ok=True)
NORM_PATH = NORM_DIR / "ssp_tl_norm_train2336_ndrz10.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------------
    print("Loading data...")
    # Load all data
    x_train_part = MatReader(str(TRAIN_X_PATH)).read_field("train_x")
    x_test_part  = MatReader(str(TEST_X_PATH)).read_field("test_x")
    x_all = torch.cat([x_train_part, x_test_part], dim=0)

    y_train_part = MatReader(str(TRAIN_Y_PATH)).read_field("train_y")
    y_test_part  = MatReader(str(TEST_Y_PATH)).read_field("test_y")
    y_all = torch.cat([y_train_part, y_test_part], dim=0)

    # Random Split
    nt_total = x_all.shape[0]
    g = torch.Generator().manual_seed(2025)
    perm = torch.randperm(nt_total, generator=g)

    train_idx = perm[:ntrain]
    test_idx  = perm[ntrain:]

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_test  = x_all[test_idx]
    y_test  = y_all[test_idx]

    print(f"Train X shape: {x_train.shape}")
    print(f"Train Y shape: {y_train.shape}")
    print(f"Test X shape: {x_test.shape}")
    print(f"Test Y shape: {y_test.shape}")

    # (N,H,W,1)
    x_train = x_train.unsqueeze(-1)
    x_test  = x_test.unsqueeze(-1)
    H, W = x_train.shape[1], x_train.shape[2]

    # --------------------------------------------------------------
    # 2. Build scalar time channel (global continuity)
    #   train idx: random subset of 0..2919
    #   test  idx: random subset of 0..2919
    #   t = idx/(nt_total-1) in [0,1]
    # --------------------------------------------------------------
    train_global_idx = train_idx.float()
    test_global_idx  = test_idx.float()

    t_train = (train_global_idx / (nt_total - 1.0)).view(-1, 1, 1, 1).repeat(1, H, W, 1)  # (N,H,W,1)
    t_test  = (test_global_idx  / (nt_total - 1.0)).view(-1, 1, 1, 1).repeat(1, H, W, 1)

    # concatenate: SSP(1) + t(1) => 2 channels
    x_train = torch.cat([x_train, t_train], dim=-1)  # (N,H,W,2)
    x_test  = torch.cat([x_test,  t_test ], dim=-1)

    # --------------------------------------------------------------
    # 3. Normalization (SSP only, time kept raw)
    # --------------------------------------------------------------
    if NORM_PATH.exists():
        print(f"Loading normalizers from {NORM_PATH}")
        norm_dict = torch.load(str(NORM_PATH), map_location="cpu")

        x_normalizer = UnitGaussianNormalizer(torch.zeros(1))  # dummy init
        x_normalizer.mean = norm_dict["x_mean"]
        x_normalizer.std  = norm_dict["x_std"]
        x_normalizer.eps  = norm_dict["eps"]

        y_normalizer = UnitGaussianNormalizer(torch.zeros(1))  # dummy init
        y_normalizer.mean = norm_dict["y_mean"]
        y_normalizer.std  = norm_dict["y_std"]
        y_normalizer.eps  = norm_dict["eps"]
    else:
        print("Computing normalizers from training data (SSP channel only)...")
        x_normalizer = UnitGaussianNormalizer(x_train[..., 0:1])  # SSP only
        y_normalizer = UnitGaussianNormalizer(y_train)

        print(f"Saving normalizers to {NORM_PATH}")
        torch.save({
            "x_mean": x_normalizer.mean.cpu(),
            "x_std":  x_normalizer.std.cpu(),
            "y_mean": y_normalizer.mean.cpu(),
            "y_std":  y_normalizer.std.cpu(),
            "eps": x_normalizer.eps,
            "meta": {
                "ntrain": int(ntrain),
                "ntest": int(ntest),
                "note": "Normalizer computed from TRAIN split only. SSP normalized; TL normalized.",
            }
        }, str(NORM_PATH))

    # Apply: normalize SSP only (channel 0)
    x_train[..., 0:1] = x_normalizer.encode(x_train[..., 0:1])
    x_test[...,  0:1] = x_normalizer.encode(x_test[...,  0:1])

    # Normalize Y (train only)
    y_train = y_normalizer.encode(y_train)

    if torch.cuda.is_available():
        x_normalizer.cuda()
        y_normalizer.cuda()

    # --------------------------------------------------------------
    # 4. Data Loaders
    # --------------------------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),  # y_test kept physical
        batch_size=batch_size, shuffle=False
    )

    # --------------------------------------------------------------
    # 5. Model Setup
    #   Input channels = 2 (SSP + time scalar)
    # --------------------------------------------------------------
    model = FNO2d(modes1, modes2, width, in_dim=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    train_loss_func = H1Loss(d=2, beta=0.01)

    # --------------------------------------------------------------
    # 6. Training Loop (print OptLoss / TrainL2 / TestL2)
    # --------------------------------------------------------------
    print("Starting training...")
    t0 = default_timer()

    train_loss_history = []
    test_loss_history = []

    for ep in range(epochs):
        model.train()
        t1 = default_timer()

        train_l2 = 0.0
        train_opt_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x).squeeze()  # (B,H,W)

            loss = train_loss_func(out, y)  # optimization loss on normalized space
            loss.backward()
            optimizer.step()

            train_opt_loss += loss.item()

            with torch.no_grad():
                out_phys = y_normalizer.decode(out)
                y_phys   = y_normalizer.decode(y)
                bsz = x.size(0)
                train_l2 += myloss(out_phys.view(bsz, -1), y_phys.view(bsz, -1)).item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x).squeeze()               # normalized pred
                out_phys = y_normalizer.decode(out)    # physical pred

                bsz = x.size(0)
                test_l2 += myloss(out_phys.view(bsz, -1), y.view(bsz, -1)).item()

        # average per sample (match your existing style)
        train_l2 /= ntrain
        test_l2  /= ntest
        train_opt_loss /= ntrain

        train_loss_history.append(train_l2)
        test_loss_history.append(test_l2)

        t2 = default_timer()
        print(f"Epoch: {ep}, Time: {t2-t1:.2f}, Opt Loss: {train_opt_loss:.5f}, Train L2: {train_l2:.5f}, Test L2: {test_l2:.5f}")

    print(f"Training completed in {default_timer()-t0:.2f}s")

    # --------------------------------------------------------------
    # 7. Save model
    # --------------------------------------------------------------
    model_path = RESULT_DIR / "model_ssp_tl_time_scalar.pth"
    torch.save(model.state_dict(), str(model_path))
    print(f"Model saved to {model_path}")

    # --------------------------------------------------------------
    # 8. Visualization (loss curve + 3 examples)
    # --------------------------------------------------------------
    print("Generating visualizations...")

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(test_loss_history, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Relative L2 Loss")
    plt.legend()
    plt.title("Training and Testing Loss")
    plt.savefig(str(RESULT_DIR / "loss_curve.png"))
    plt.close()

    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        x, y = x.to(device), y.to(device)

        out = model(x).squeeze()
        out = y_normalizer.decode(out)  # physical prediction

        # decode SSP for plotting
        ssp_norm = x[..., 0:1]
        ssp_decoded = x_normalizer.decode(ssp_norm)
        x_ssp = ssp_decoded[..., 0]

        fig = plt.figure(figsize=(15, 12))
        n_examples = 3

        for idx in range(n_examples):
            # scalar time channel at [idx,0,0,1]
            t_val = x[idx, 0, 0, 1].item()
            day = t_val * 365.0

            ssp = x_ssp[idx].cpu().numpy()
            y_true = y[idx].cpu().numpy()
            y_pred = out[idx].cpu().numpy()
            error = y_pred - y_true

            ax = fig.add_subplot(n_examples, 4, idx*4 + 1)
            im1 = ax.imshow(ssp, cmap="viridis", aspect="auto")
            if idx == 0:
                ax.set_title(f"Input SSP (day {day:.1f})", fontsize=12)
            plt.colorbar(im1, ax=ax, shrink=0.8)

            ax = fig.add_subplot(n_examples, 4, idx*4 + 2)
            im2 = ax.imshow(y_true, cmap="turbo", aspect="auto", vmin=40, vmax=110)
            if idx == 0:
                ax.set_title("Ground Truth TL", fontsize=12)
            plt.colorbar(im2, ax=ax, shrink=0.8)

            ax = fig.add_subplot(n_examples, 4, idx*4 + 3)
            im3 = ax.imshow(y_pred, cmap="turbo", aspect="auto", vmin=40, vmax=110)
            if idx == 0:
                ax.set_title("FNO Prediction", fontsize=12)
            plt.colorbar(im3, ax=ax, shrink=0.8)

            ax = fig.add_subplot(n_examples, 4, idx*4 + 4)
            error_max = max(abs(error.min()), abs(error.max()))
            im4 = ax.imshow(error, cmap="RdBu_r", aspect="auto", vmin=-error_max, vmax=error_max)
            if idx == 0:
                ax.set_title("Error", fontsize=12)
            plt.colorbar(im4, ax=ax, shrink=0.8)

            rmse = np.sqrt(np.mean(error**2))
            ax.text(
                0.02, 0.98, f"RMSE: {rmse:.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

        plt.suptitle("SSP to TL Prediction Results (Time Scalar)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_fig = RESULT_DIR / "prediction_results.png"
        plt.savefig(str(out_fig))
        plt.close()
        print(f"Saved visualization to {out_fig}")

    # 9. Full Test Set RMSE Calculation
    print("Calculating full test set metrics...")
    model.eval()
    sqerr_sum = 0.0
    count = 0
    rmse_list = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze()
            pred = y_normalizer.decode(pred)

            err = pred - y                      # (B, H, W)
            mse_per = (err**2).mean(dim=(1,2))  # per-sample MSE
            rmse_per = torch.sqrt(mse_per)

            rmse_list.append(rmse_per.cpu())
            sqerr_sum += (err**2).sum().item()
            count += err.numel()

    rmse_all = (sqerr_sum / count) ** 0.5
    rmse_per_all = torch.cat(rmse_list)

    print(
        f"TEST_RMSE_ALLPOINTS={rmse_all:.4f}, "
        f"MEAN_PER_SAMPLE={rmse_per_all.mean():.4f}, "
        f"STD_PER_SAMPLE={rmse_per_all.std():.4f}"
    )


if __name__ == "__main__":
    main()
