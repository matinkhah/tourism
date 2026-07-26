# preprocess.py

import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# Dataset constants (Section IV-A)
# ---------------------------------------------------------------------------
DATASET_START = "2009-01-01"
DATASET_END = "2016-12-31"          # 2,922 total daily observations
TRAIN_END_DATE = "2014-12-07"       # 2,167 days -> 74.2%
VAL_END_DATE = "2015-09-25"         # +292 days  -> 10.0%
# remaining 463 days (2015-09-26 .. 2016-12-31) form the 15.8% test split

# The 16 meteorological attributes of Equation (3), in the exact order
# W_t = [T, Tpot, Tdew, RH, VPmax, VPact, VPdef, sh, H2OC, p, rho, wv,
#        wmax, wd, R, Rdur]^T
JENA_AGG_RULES = {
    "p (mbar)": "mean",          # Total Atmospheric Pressure (p)
    "T (degC)": "mean",          # Surface Air Temperature (T)
    "Tpot (K)": "mean",          # Potential Temperature (Tpot)
    "Tdew (degC)": "mean",       # Dew Point Temperature (Tdew)
    "rh (%)": "mean",            # Relative Humidity (RH)
    "VPmax (mbar)": "mean",      # Saturation Vapor Pressure (VPmax)
    "VPact (mbar)": "mean",      # Actual Vapor Pressure (VPact)
    "VPdef (mbar)": "mean",      # Vapor Pressure Deficit (VPdef)
    "sh (g/kg)": "mean",         # Specific Humidity (sh)
    "H2OC (mmol/mol)": "mean",   # Water Vapor Concentration (H2OC)
    "rho (g/m**3)": "mean",      # Air Density (rho)
    "wv (m/s)": "mean",          # Wind Velocity (wv)
    "max. wv (m/s)": "max",      # Peak Wind Gust Velocity (wmax)
    "wd (deg)": "mean",          # Wind Direction (wd)
    "rain (mm)": "sum",          # Cumulative Precipitation Depth (R)
    "raining (s)": "sum",        # Active Rainfall Duration (Rdur)
}


def enforce_determinism(seed):
    """
    Enforces absolute determinism across all framework RNGs as specified
    in Section V ("Framework-level deterministic modes were strictly
    enforced by setting torch.backends.cudnn.deterministic = True and
    torch.backends.cudnn.benchmark = False").
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def aggregate_jena_to_daily(raw_jena_path):
    """
    Aggregates 10-minute Beutenberg station entries into the 16 daily
    climate metrics of Equation (3). Thermodynamic/vapor/kinematic
    states are averaged; precipitation quantities are summed; peak gust
    is the daily maximum.
    """
    df = pd.read_csv(raw_jena_path)
    df["date"] = pd.to_datetime(df["date"])

    existing_rules = {k: v for k, v in JENA_AGG_RULES.items() if k in df.columns}
    missing = [k for k in JENA_AGG_RULES if k not in df.columns]
    if missing:
        print(f"Warning: {len(missing)} of the 16 climate columns from Eq.(3) "
              f"are missing from the raw file and will be skipped: {missing}")

    df_daily = df.resample("D", on="date").agg(existing_rules).reset_index()
    df_daily = df_daily.set_index("date").reindex(
        pd.date_range(DATASET_START, DATASET_END, freq="D")
    )
    df_daily = df_daily.interpolate(limit_direction="both").reset_index()
    df_daily = df_daily.rename(columns={"index": "date"})
    return df_daily


def spline_monthly_tourism_to_daily(tourism_csv_path, target_col="tourist_count"):
    """
    Converts the monthly Thuringia hospitality registry (Section IV-A,
    Table ge000802) into a smooth daily latent-demand trajectory using a
    *monotonic* cubic Hermite interpolator (PCHIP). Per Section III-B,
    this behaves as a deterministic low-pass filter: it removes
    localized hotel-specific booking-lag / weekend-batching noise while
    preserving the continuous macro-demand baseline that the model is
    actually asked to forecast.
    """
    df_monthly = pd.read_csv(tourism_csv_path)
    df_monthly["date"] = pd.to_datetime(df_monthly["date"])
    df_monthly = df_monthly.sort_values("date").drop_duplicates("date")

    daily_index = pd.date_range(DATASET_START, DATASET_END, freq="D")

    # PCHIP requires a strictly increasing x-axis expressed as a numeric
    # ordinate (e.g. days since the first monthly reading).
    x_monthly = (df_monthly["date"] - df_monthly["date"].iloc[0]).dt.days.values
    y_monthly = df_monthly[target_col].values
    interpolator = PchipInterpolator(x_monthly, y_monthly, extrapolate=True)

    x_daily = (daily_index - df_monthly["date"].iloc[0]).days.values
    y_daily = interpolator(x_daily)

    df_daily = pd.DataFrame({"date": daily_index, target_col: y_daily})
    return df_daily


def generate_sliding_windows(df, seq_len=96, pred_len=24, target_col="tourist_count"):
    """
    Transforms a clean, multivariate continuous timeline into supervised
    (look-back, horizon) matrix pairs per Equation (4):
        X_t-L+1:t  ->  Y_t+1:t+H
    The target column is enforced to sit at the final channel index (OT
    convention), matching Equation (2)'s ordering X_t = [y_t, w_t^1, ...].
    """
    cols = [c for c in df.columns if c not in ("date", target_col)] + [target_col]
    data_matrix = df[cols].values.astype(np.float32)

    X, Y = [], []
    for i in range(len(data_matrix) - seq_len - pred_len + 1):
        X.append(data_matrix[i: i + seq_len, :])
        Y.append(data_matrix[i + seq_len: i + seq_len + pred_len, :])

    return np.array(X), np.array(Y)


def prepare_pipeline(
    raw_jena_path,
    tourism_csv_path,
    seq_len=96,
    pred_len=24,
    target_col="tourist_count",
    train_end_date=TRAIN_END_DATE,
    val_end_date=VAL_END_DATE,
):
    """
    Complete data pipeline matching Sections III-A and IV-A:
      1) aggregate the 16-variable microclimate matrix to daily
      2) spline-interpolate the monthly tourism target to daily
      3) merge into the 17-channel array
      4) chronological split: 2,167 / 292 / 463 days
      5) Z-score fit strictly on the training partition
      6) sliding-window supervised generation (L=96, H in {24,48,96})
    """
    # 1. Temporal Resolution Alignment Block (16 climate variables)
    df_climate = aggregate_jena_to_daily(raw_jena_path)

    # 2. Target Feature Construction via monotonic cubic-spline
    #    interpolation of the monthly Thuringia registry
    if tourism_csv_path and os.path.exists(tourism_csv_path):
        df_tourism = spline_monthly_tourism_to_daily(tourism_csv_path, target_col)
    else:
        raise FileNotFoundError(
            f"Tourism target file not found at '{tourism_csv_path}'. "
            "The paper's target vector is derived exclusively from the "
            "official Thuringia statistical registry (Table ge000802) "
            "via monotonic cubic-spline interpolation; no synthetic "
            "fallback is used in the reproduction pipeline."
        )

    # 3. Merge the 16-dim climate matrix with the 1-dim spline target
    #    into the unified 17-dimensional channel array (Algorithm 1, L2)
    df = pd.merge(df_climate, df_tourism, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    n_total = len(df)
    expected_days = (pd.Timestamp(DATASET_END) - pd.Timestamp(DATASET_START)).days + 1
    if n_total != expected_days:
        print(f"Warning: merged series has {n_total} days; Section IV-A reports "
              f"{expected_days} days ({DATASET_START} to {DATASET_END}).")

    # 4. Chronological Train/Val/Test Split using the exact date
    #    boundaries reported in Section IV-A (74.2% / 10.0% / 15.8%)
    train_mask = df["date"] <= pd.Timestamp(train_end_date)
    val_mask = (df["date"] > pd.Timestamp(train_end_date)) & (df["date"] <= pd.Timestamp(val_end_date))
    test_mask = df["date"] > pd.Timestamp(val_end_date)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    # 5. Standard Scaler Configuration (Z-score fit strictly on training
    #    matrix to prevent validation/test-bias contamination)
    numeric_cols = [c for c in df.columns if c != "date"]
    means = train_df[numeric_cols].mean()
    stds = train_df[numeric_cols].std()

    for partition in (train_df, val_df, test_df):
        partition[numeric_cols] = (partition[numeric_cols] - means) / (stds + 1e-8)

    # 6. Sliding-Window Supervised Generation (L=96, H in {24, 48, 96})
    X_train, Y_train = generate_sliding_windows(train_df, seq_len, pred_len, target_col)
    X_val, Y_val = generate_sliding_windows(val_df, seq_len, pred_len, target_col)
    X_test, Y_test = generate_sliding_windows(test_df, seq_len, pred_len, target_col)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, means, stds


if __name__ == "__main__":
    # Reproducibility verification test
    enforce_determinism(42)
    print("Determinism successfully mapped to GPU frameworks.")
    print("Pipeline ready to execute with unified 17-channel climate + "
          "spline-interpolated tourism aggregation.")
