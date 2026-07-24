import os
import random
import numpy as np
import pandas as pd
import torch

def enforce_determinism(seed):
    """Enforces absolute determinism across all framework RNGs as specified in Section IV."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_sliding_windows(df, seq_len=96, pred_len=30, target_col='tourist_count'):
    """Transforms raw multivariate matrix into chronological supervised pairs."""
    # Ensure target column is the absolute last column (OT configuration)
    cols = [c for c in df.columns if c != 'date' and c != target_col] + [target_col]
    data_matrix = df[cols].values
    
    X, Y = [], []
    for i in range(len(data_matrix) - seq_len - pred_len + 1):
        X.append(data_matrix[i : i + seq_len, :])
        # Target sequence extracts future horizons for backpropagation mapping
        Y.append(data_matrix[i + seq_len : i + seq_len + pred_len, :])
        
    return np.array(X), np.array(Y)

def prepare_pipeline(csv_path, seq_len=96, pred_len=30):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Chronological Split (70/10/20) preventing future information leakage
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Fit Z-score normalization scaling strictly on the training partition matrix
    numeric_cols = [c for c in df.columns if c != 'date']
    means = train_df[numeric_cols].mean()
    stds = train_df[numeric_cols].std()
    
    for d in [train_df, val_df, test_df]:
        d[numeric_cols] = (d[numeric_cols] - means) / (stds + 1e-8)
        
    X_train, Y_train = generate_sliding_windows(train_df, seq_len, pred_len)
    X_val, Y_val = generate_sliding_windows(val_df, seq_len, pred_len)
    X_test, Y_test = generate_sliding_windows(test_df, seq_len, pred_len)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, means, stds

if __name__ == "__main__":
    enforce_determinism(42) # First experimental execution seed
    print("Determinism successfully mapped to GPU frameworks.")
