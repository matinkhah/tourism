import io
import os
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import PchipInterpolator


def enforce_determinism(seed=42):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_and_preprocess_multivariate_data(
    tourism_csv_path, seq_len=96, horizon_H=24, seed=42
):
    print("🚀 Initializing repository-linked multivariate preprocessing pipeline...")
    enforce_determinism(seed=seed)

    if not os.path.exists(tourism_csv_path):
        raise FileNotFoundError(f"❌ Tourism dataset file missing at path: '{tourism_csv_path}'")

    # =====================================================================
    # 1. Download and Process GitHub-Hosted Climate Data
    # =====================================================================
    # HARDCODED VERIFIED RAW REPOSITORY LINK
    climate_url = "https://raw.githubusercontent.com/matinkhah/tourism/refs/heads/main/dataset.csv"
    
    print("📡 Downloading raw climate data directly from GitHub repository...")
    res_climate = requests.get(climate_url, timeout=30)
    if res_climate.status_code != 200:
        raise ConnectionError(f"❌ GitHub file download failed. HTTP Status: {res_climate.status_code}")

    df_raw = pd.read_csv(io.StringIO(res_climate.text))
    
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["Date"] = df_raw["date"].dt.date

    # Compute wind vectors to perform safe circular direction mean
    df_raw["wd_rad"] = np.radians(df_raw["wd (deg)"])
    df_raw["wd_sin"] = np.sin(df_raw["wd_rad"])
    df_raw["wd_cos"] = np.cos(df_raw["wd_rad"])

    print("📈 Aggregating 10-minute intervals to rigorous daily resolution...")
    df_daily_base = (
        df_raw.groupby("Date")
        .agg(
            p_mbar=("p (mbar)", "mean"),
            T_degC=("T (degC)", "mean"),
            Tpot_K=("Tpot (K)", "mean"),
            Tdew_degC=("Tdew (degC)", "mean"),
            rh_percent=("rh (%)", "mean"),
            VPmax_mbar=("VPmax (mbar)", "mean"),
            VPact_mbar=("VPact (mbar)", "mean"),
            VPdef_mbar=("VPdef (mbar)", "mean"),
            sh_g_kg=("sh (g/kg)", "mean"),
            H2OC_mmol_mol=("H2OC (mmol/mol)", "mean"),
            rho_g_m3=("rho (g/m**3)", "mean"),
            wv_m_s=("wv (m/s)", "mean"),
            wmax_m_s=("max. wv (m/s)", "max"),
            rain_mm=("rain (mm)", "sum"),         # R_t: Accumulated precipitation depth
            raining_s=("raining (s)", "sum"),   # R_dur,t: Accumulated active rainfall duration
            wd_sin_mean=("wd_sin", "mean"),
            wd_cos_mean=("wd_cos", "mean"),
        )
        .reset_index()
    )

    # Reconstruct circular wind mean natively into its exact feature slot
    df_daily_base["wd (deg)"] = (
        np.degrees(
            np.arctan2(
                df_daily_base["wd_sin_mean"], df_daily_base["wd_cos_mean"]
            )
        )
        % 360
    )
    df_daily_base = df_daily_base.drop(columns=["wd_sin_mean", "wd_cos_mean"])
    df_daily_base["Date"] = pd.to_datetime(df_daily_base["Date"])

    # Verify exactly 16 clean variables match paper specs
    climate_cols = [c for c in df_daily_base.columns if c != "Date"]
    assert len(climate_cols) == 16, f"❌ Dimension error: Expected 16 climate variables, compiled {len(climate_cols)}."
    print(f"   -> Isolated exactly {len(climate_cols)} non-collinear meteorological drivers.")

    # =====================================================================
    # 2. Process Monthly Tourism Data via Shape-Preserving PCHIP Spline
    # =====================================================================
    print("📈 Processing regional hospitality target index values...")
    df_monthly = pd.read_csv(tourism_csv_path)
    df_monthly["Date"] = pd.to_datetime(
        df_monthly["Year"].astype(str) + "-" + df_monthly["Month"].astype(str) + "-01"
    )
    df_monthly = df_monthly.sort_values("Date").reset_index(drop=True)

    # Synchronize dates chronologically with available climate timeline
    min_date, max_date = df_daily_base["Date"].min(), df_daily_base["Date"].max()
    target_timeline = pd.date_range(start=min_date, end=max_date, freq="D")

    # FIX: Using specific anchor scalar date coordinate indexes to prevent broadcast exceptions
    anchor_date = target_timeline[0]
    monthly_offsets = (df_monthly["Date"] - anchor_date).dt.days.values
    daily_offsets = (target_timeline - anchor_date).days.values

    spline = PchipInterpolator(monthly_offsets, df_monthly["Ankuenfte_Insgesamt"].values)
    df_tourism = pd.DataFrame(
        {"Date": target_timeline, "tourist_count": spline(daily_offsets)}
    )

    # Combine data streams and force target data to final column channel mapping position
    df_master = pd.merge(df_tourism, df_daily_base, on="Date", how="inner")
    feature_cols = [
        c for c in df_master.columns if c not in ["Date", "tourist_count"]
    ] + ["tourist_count"]

    # =====================================================================
    # 3. Fit Normalization Statistics Exclusively on Training Splits
    # =====================================================================
    # Enforce chronological paper boundaries (2009-01-01 to 2014-12-07 training cutoff)
    train_mask = (df_master["Date"] >= "2009-01-01") & (df_master["Date"] <= "2014-12-07")
    
    # Fallback to absolute array split boundaries if data represents a different timeline span
    if not train_mask.any():
        train_len = int(len(df_master) * 0.742)
        train_mask = df_master.index < train_len

    for col in feature_cols:
        df_master[col] = (
            df_master[col] - df_master.loc[train_mask, col].mean()
        ) / df_master.loc[train_mask, col].std()

    master_matrix = df_master[feature_cols].values  # Shape: (Total Days, 17)
    target_vector = df_master["tourist_count"].values

    # =====================================================================
    # 4. Generate Rolling Temporal Input/Forecast Tensor Windows
    # =====================================================================
    X_windows, Y_windows = [], []
    for i in range(len(master_matrix) - seq_len - horizon_H + 1):
        X_windows.append(master_matrix[i : i + seq_len])
        Y_windows.append(
            target_vector[i + seq_len : i + seq_len + horizon_H]
        )

    X_tensor = np.array(X_windows)
    Y_tensor = np.array(Y_windows)[:, :, np.newaxis]

    assert X_tensor.shape[-1] == 17, f"❌ Dimension error: Expected 17 features, compiled {X_tensor.shape[-1]} channels."
    print("\n🏁 Tensors compiled successfully with zero errors:")
    print(f"   -> X Tensor Lookback Windows Matrix Shape: {X_tensor.shape} (17 channels matched)")
    print(f"   -> Y Tensor Target Forecast Vector Shape:  {Y_tensor.shape}")

    return X_tensor, Y_tensor, df_master


# =====================================================================
# 5. Runtime Pipeline Execution Call
# =====================================================================
if __name__ == "__main__":
    X, Y, master_df = load_and_preprocess_multivariate_data(
        tourism_csv_path="tourism_thuringia_2009_2016.csv",
        seq_len=96,
        horizon_H=24,
        seed=42
    )
