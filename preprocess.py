import io
import os
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import PchipInterpolator


def enforce_determinism(seed: int = 42) -> None:
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
    tourism_csv_path: str,
    seq_len: int = 96,
    horizon_H: int = 24,
    seed: int = 42,
    climate_csv_path: str = "dataset.csv",
):
    print("🚀 Initializing multivariate preprocessing pipeline...")
    enforce_determinism(seed=seed)

    if not os.path.exists(tourism_csv_path):
        raise FileNotFoundError(f"❌ Tourism file missing: '{tourism_csv_path}'")

    # =====================================================================
    # 1. Load climate data (local file preferred)
    # =====================================================================
    print("📡 Loading climate data...")
    if os.path.exists(climate_csv_path):
        df_raw = pd.read_csv(climate_csv_path)
        print(f"   -> Loaded local file: {climate_csv_path}")
    else:
        # Fallback to the old GitHub-hosted file
        climate_url = "https://raw.githubusercontent.com/matinkhah/tourism/refs/heads/main/dataset.csv"
        print("   -> Local file not found, downloading from GitHub...")
        res = requests.get(climate_url, timeout=60)
        if res.status_code != 200:
            raise ConnectionError(f"❌ Download failed (HTTP {res.status_code})")
        df_raw = pd.read_csv(io.StringIO(res.text))

    # --- Normalise datetime column name ---
    if "Date Time" in df_raw.columns:
        df_raw["date"] = pd.to_datetime(df_raw["Date Time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    elif "date" in df_raw.columns:
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    else:
        raise KeyError("Could not find a datetime column ('Date Time' or 'date')")

    df_raw = df_raw.dropna(subset=["date"]).copy()
    df_raw["Date"] = df_raw["date"].dt.normalize()   # midnight timestamps

    # --- Circular wind direction helpers ---
    if "wd (deg)" in df_raw.columns:
        df_raw["wd_rad"] = np.radians(df_raw["wd (deg)"])
        df_raw["wd_sin"] = np.sin(df_raw["wd_rad"])
        df_raw["wd_cos"] = np.cos(df_raw["wd_rad"])
    else:
        df_raw["wd_sin"] = 0.0
        df_raw["wd_cos"] = 1.0

    print("📈 Aggregating 10-minute data to daily resolution...")

    # Build aggregation dictionary only for columns that actually exist
    agg_dict = {}
    candidates = {
        "p_mbar":          ("p (mbar)", "mean"),
        "T_degC":          ("T (degC)", "mean"),
        "Tpot_K":          ("Tpot (K)", "mean"),
        "Tdew_degC":       ("Tdew (degC)", "mean"),
        "rh_percent":      ("rh (%)", "mean"),
        "VPmax_mbar":      ("VPmax (mbar)", "mean"),
        "VPact_mbar":      ("VPact (mbar)", "mean"),
        "VPdef_mbar":      ("VPdef (mbar)", "mean"),
        "sh_g_kg":         ("sh (g/kg)", "mean"),
        "H2OC_mmol_mol":   ("H2OC (mmol/mol)", "mean"),
        "rho_g_m3":        ("rho (g/m**3)", "mean"),
        "wv_m_s":          ("wv (m/s)", "mean"),
        "wmax_m_s":        ("max. wv (m/s)", "max"),
        "rain_mm":         ("rain (mm)", "sum"),
        "raining_s":       ("raining (s)", "sum"),
    }

    for new_name, (old_name, how) in candidates.items():
        if old_name in df_raw.columns:
            agg_dict[new_name] = (old_name, how)

    # Always aggregate the wind vectors if present
    if "wd_sin" in df_raw.columns:
        agg_dict["wd_sin_mean"] = ("wd_sin", "mean")
        agg_dict["wd_cos_mean"] = ("wd_cos", "mean")

    df_daily = (
        df_raw.groupby("Date")
        .agg(**{k: v for k, v in agg_dict.items()})
        .reset_index()
    )

    # Reconstruct mean wind direction
    if "wd_sin_mean" in df_daily.columns:
        df_daily["wd_deg"] = (
            np.degrees(np.arctan2(df_daily["wd_sin_mean"], df_daily["wd_cos_mean"])) % 360
        )
        df_daily = df_daily.drop(columns=["wd_sin_mean", "wd_cos_mean"])
    else:
        df_daily["wd_deg"] = 0.0

    df_daily["Date"] = pd.to_datetime(df_daily["Date"])

    climate_cols = [c for c in df_daily.columns if c != "Date"]
    print(f"   -> Daily climate variables available: {len(climate_cols)} → {climate_cols}")

    # =====================================================================
    # 2. Monthly tourism → daily via PCHIP
    # =====================================================================
    print("📈 Processing tourism target (PCHIP interpolation)...")
    df_monthly = pd.read_csv(tourism_csv_path)
    df_monthly["Date"] = pd.to_datetime(
        df_monthly["Year"].astype(str) + "-" + df_monthly["Month"].astype(str) + "-01"
    )
    df_monthly = df_monthly.sort_values("Date").reset_index(drop=True)

    min_date = df_daily["Date"].min()
    max_date = df_daily["Date"].max()
    target_timeline = pd.date_range(start=min_date, end=max_date, freq="D")

    anchor = target_timeline[0]
    monthly_offsets = (df_monthly["Date"] - anchor).dt.days.values
    daily_offsets   = (target_timeline - anchor).days.values

    spline = PchipInterpolator(monthly_offsets, df_monthly["Ankuenfte_Insgesamt"].values)
    df_tourism = pd.DataFrame({
        "Date": target_timeline,
        "tourist_count": spline(daily_offsets)
    })

    # Inner join keeps only overlapping days
    df_master = pd.merge(df_tourism, df_daily, on="Date", how="inner")
    feature_cols = [c for c in df_master.columns if c not in ["Date", "tourist_count"]] + ["tourist_count"]

    print(f"   -> Merged timeline: {df_master['Date'].min().date()} → {df_master['Date'].max().date()} "
          f"({len(df_master)} days)")

    # =====================================================================
    # 3. Train-only Z-score normalisation
    # =====================================================================
    train_mask = (df_master["Date"] >= "2009-01-01") & (df_master["Date"] <= "2014-12-07")
    if not train_mask.any():
        # Fallback if the climate file has a different date range
        train_len = int(len(df_master) * 0.742)
        train_mask = df_master.index < train_len
        print("   ⚠ Using percentage-based train split (date boundaries not found)")

    for col in feature_cols:
        mu = df_master.loc[train_mask, col].mean()
        sigma = df_master.loc[train_mask, col].std()
        df_master[col] = (df_master[col] - mu) / (sigma + 1e-8)

    master_matrix = df_master[feature_cols].values
    target_vector = df_master["tourist_count"].values

    # =====================================================================
    # 4. Sliding windows
    # =====================================================================
    X_windows, Y_windows = [], []
    for i in range(len(master_matrix) - seq_len - horizon_H + 1):
        X_windows.append(master_matrix[i : i + seq_len])
        Y_windows.append(target_vector[i + seq_len : i + seq_len + horizon_H])

    X_tensor = np.asarray(X_windows, dtype=np.float32)
    Y_tensor = np.asarray(Y_windows, dtype=np.float32)[:, :, np.newaxis]

    n_features = X_tensor.shape[-1]
    print("\n🏁 Preprocessing finished:")
    print(f"   -> X shape : {X_tensor.shape}  (features = {n_features})")
    print(f"   -> Y shape : {Y_tensor.shape}")
    print(f"   -> Feature order: {feature_cols}")

    return X_tensor, Y_tensor, df_master


if __name__ == "__main__":
    X, Y, master_df = load_and_preprocess_multivariate_data(
        tourism_csv_path="tourism_thuringia_2009_2016.csv",
        seq_len=96,
        horizon_H=24,
        seed=42,
    )
