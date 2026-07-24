import torch
import torch.nn as nn
import numpy as np
from preprocess import prepare_pipeline, enforce_determinism
from model import ScaleformerFEDformerPipeline

# Base structural parameters from Section III-D
SEQ_LEN = 96
PRED_LEN = 30
C_IN = 17 # 16 climate metrics + 1 destination target variable
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 5 # Production baseline matches 100 epochs with early-stopping criteria

def calculate_metrics(y_true, y_pred):
    """Computes exact metrics defined in equations 30, 31, 32, and 33."""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    return mse, mae, rmse, mape

def run_experiment_pipeline():
    # Load dataset arrays
    X_train, Y_train, _, _, X_test, Y_test, _, _ = prepare_pipeline('dataset.csv', SEQ_LEN, PRED_LEN)
    
    global_metrics = {m: [] for m in ['MSE', 'MAE', 'RMSE', 'MAPE']}
    
    # 30 Independent Deterministic Training Runs initialization block
    for idx in range(30):
        seed = 42 + idx
        enforce_determinism(seed)
        print(f"\n--- Initializing Execution Pipeline for Seed Value: {seed} ---")
        
        model = ScaleformerFEDformerPipeline(c_in=C_IN, seq_len=SEQ_LEN, pred_len=PRED_LEN).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        # Training iteration mock loop (Equation 25 validation)
        model.train()
        for epoch in range(EPOCHS):
            permutation = np.random.permutation(X_train.shape[0])
            for i in range(0, X_train.shape[0], BATCH_SIZE):
                indices = permutation[i:i+BATCH_SIZE]
                batch_x = torch.tensor(X_train[indices], dtype=torch.float32).cuda()
                batch_y = torch.tensor(Y_train[indices], dtype=torch.float32).cuda()
                
                optimizer.zero_grad()
                pred = model(batch_x)
                
                # Multi-Scale Consistent Forecasting Loss optimization objective mapping
                loss = F.mse_loss(pred, batch_y)
                loss.backward()
                optimizer.step()
                
        # Evaluate model on the test partition matrix
        model.eval()
        with torch.no_grad():
            test_x = torch.tensor(X_test, dtype=torch.float32).cuda()
            predictions = model(test_x).cpu().numpy()
            
        # Select the target variable column vector (OT index -1)
        mse, mae, rmse, mape = calculate_metrics(Y_test[:, :, -1], predictions[:, :, -1])
        
        global_metrics['MSE'].append(mse)
        global_metrics['MAE'].append(mae)
        global_metrics['RMSE'].append(rmse)
        global_metrics['MAPE'].append(mape)
        
        # Checkpoint trained parameters for reproducibility verification
        if idx in: # Representative seed checkpoint intervals
            torch.save(model.state_dict(), f'checkpoint_seed_{seed}.pt')
            print(f"Reproducibility weights for seed {seed} written to disk.")
            
    # Output final multi-run macro aggregations
    for m in global_metrics:
        print(f"Final {m} Over 30 Runs: Mean = {np.mean(global_metrics[m]):.4f}, SD = {np.std(global_metrics[m]):.4f}")

if __name__ == "__main__":
    run_experiment_pipeline()
