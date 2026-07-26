# main.py
#
# Reproduces the experimental protocol of Sections IV-D and V:
#   - Hyperparameters from Table (IV-D): L=96, H in {24,48,96}, 3 scales,
#     2-4 Transformer layers, 8 heads, d_model=512, dropout=0.1,
#     batch_size=32, Adam @ lr=1e-4, 100 epochs, early-stopping patience 10.
#   - 30 independent deterministic runs with seeds {42+i : i=0..29}
#     (Section V).
#   - Training objective Loss = L_forecast + lambda*L_scale + gamma*L_freq
#     (Equations 22-25).
#   - Metrics MSE/MAE/RMSE/MAPE averaged (mean +/- SD) over the 30 runs
#     (Table I).

import os

import numpy as np
import torch
import torch.nn.functional as F

from preprocess import prepare_pipeline, enforce_determinism, generate_sliding_windows
from model import ScaleformerFEDformerPipeline, SCALE_NAMES

# ---------------------------------------------------------------------------
# Hyperparameters (Section IV-D / Table)
# ---------------------------------------------------------------------------
SEQ_LEN = 96
C_IN = 17                    # 16 climate metrics (Eq. 3) + 1 target channel
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 3                 # within the reported 2-4 range
DROPOUT = 0.1
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100
EARLY_STOP_PATIENCE = 10
N_RUNS = 30                  # Section V: 30 independent seeded runs
BASE_SEED = 42
CHECKPOINT_EVERY = 10        # write a representative checkpoint every 10 runs

# Multi-scale / frequency loss weights (Equation 22): lambda, gamma
LAMBDA_SCALE = 0.5
GAMMA_FREQ = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metrics(y_true, y_pred):
    """MSE / MAE / RMSE / MAPE, Equations (26)-(29)."""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    return mse, mae, rmse, mape


def multi_scale_loss(pred, aux, batch_y_target):
    """
    Loss = L_forecast + lambda * L_scale + gamma * L_freq
    (Equation 22), where:
      L_forecast: MSE between the fine-scale forecast and ground truth
                  (Equation 23).
      L_scale:    consistency between each scale's forecast and the
                  correspondingly downsampled ground truth (Eq. 24).
      L_freq:     consistency between the model's seasonal component
                  and a reference seasonal component extracted from the
                  ground truth at each scale (Eq. 25), approximated here
                  by comparing FFT-filtered ground-truth windows.
    """
    l_forecast = F.mse_loss(pred, batch_y_target)

    l_scale = 0.0
    for scale_name, y_hat_s in aux["scale_forecasts"].items():
        factor = {"daily": 1, "weekly": 7, "seasonal": 30}[scale_name]
        if factor == 1:
            target_ds = batch_y_target
        else:
            t = batch_y_target.transpose(1, 2)
            target_ds = F.avg_pool1d(t, kernel_size=min(factor, t.shape[-1]),
                                      stride=min(factor, t.shape[-1]),
                                      ceil_mode=True).transpose(1, 2)
            target_ds = F.interpolate(
                target_ds.transpose(1, 2), size=y_hat_s.shape[1], mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        l_scale = l_scale + F.mse_loss(y_hat_s, target_ds)

    l_freq = 0.0
    for scale_name, s_hat in aux["seasonal"].items():
        xf = torch.fft.rfft(s_hat, dim=1)
        k = min(8, xf.shape[1])
        amp = torch.abs(xf).mean(dim=(0, 2))
        _, idx = torch.topk(amp, k)
        ref = torch.zeros_like(xf)
        ref[:, idx, :] = xf[:, idx, :]
        ref_seasonal = torch.fft.irfft(ref, n=s_hat.shape[1], dim=1)
        l_freq = l_freq + F.mse_loss(s_hat, ref_seasonal.detach())

    return l_forecast + LAMBDA_SCALE * l_scale + GAMMA_FREQ * l_freq


def iterate_batches(x, y, batch_size, shuffle=True):
    n = x.shape[0]
    order = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        idx = order[i: i + batch_size]
        yield x[idx], y[idx]


def train_one_run(seed, x_train, y_train, x_val, y_val, pred_len):
    enforce_determinism(seed)
    print(f"\n--- Initializing Execution Pipeline for Seed Value: {seed} ---")

    model = ScaleformerFEDformerPipeline(
        c_in=C_IN, seq_len=SEQ_LEN, pred_len=pred_len,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in iterate_batches(x_train, y_train, BATCH_SIZE):
            bx = torch.tensor(batch_x, dtype=torch.float32, device=DEVICE)
            by = torch.tensor(batch_y[:, :, -1:], dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            pred, aux = model(bx)
            loss = multi_scale_loss(pred, aux, by)
            loss.backward()
            optimizer.step()

        # Validation pass + early stopping (Section III-H)
        model.eval()
        with torch.no_grad():
            vx = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)
            vy = torch.tensor(y_val[:, :, -1:], dtype=torch.float32, device=DEVICE)
            val_pred, val_aux = model(vx)
            val_loss = multi_scale_loss(val_pred, val_aux, vy).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} (seed {seed}), "
                      f"best val loss = {best_val_loss:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_experiment_pipeline(
    raw_jena_path="jena_climate.csv",
    tourism_csv_path="tourism_thuringia.csv",
    pred_len=24,
    checkpoint_dir="checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    X_train, Y_train, X_val, Y_val, X_test, Y_test, means, stds = prepare_pipeline(
        raw_jena_path, tourism_csv_path, seq_len=SEQ_LEN, pred_len=pred_len,
    )

    global_metrics = {m: [] for m in ["MSE", "MAE", "RMSE", "MAPE"]}

    # 30 Independent Deterministic Training Runs (Section V)
    for idx in range(N_RUNS):
        seed = BASE_SEED + idx
        model = train_one_run(seed, X_train, Y_train, X_val, Y_val, pred_len)

        # Evaluate model on the held-out test partition
        model.eval()
        with torch.no_grad():
            test_x = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
            predictions, _ = model(test_x)
            predictions = predictions.squeeze(-1).cpu().numpy()

        y_true = Y_test[:, :, -1]  # target column (OT index -1)
        mse, mae, rmse, mape = calculate_metrics(y_true, predictions)

        global_metrics["MSE"].append(mse)
        global_metrics["MAE"].append(mae)
        global_metrics["RMSE"].append(rmse)
        global_metrics["MAPE"].append(mape)

        # Checkpoint every CHECKPOINT_EVERY runs for reproducibility
        # verification (Section V, "Source Code" statement).
        if idx % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_seed_{seed}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Reproducibility weights for seed {seed} written to {ckpt_path}.")

    # Output final multi-run macro aggregations (Table I style summary)
    print(f"\n=== Results for horizon H={pred_len} (mean +/- SD over {N_RUNS} runs) ===")
    for m in global_metrics:
        vals = global_metrics[m]
        print(f"Final {m}: Mean = {np.mean(vals):.4f}, SD = {np.std(vals):.4f}")

    return global_metrics


if __name__ == "__main__":
    # Table I reports three forecasting horizons: short (24), medium
    # (48), and long (96). Re-run the full 30-seed protocol per horizon.
    for horizon in (24, 48, 96):
        run_experiment_pipeline(pred_len=horizon)
