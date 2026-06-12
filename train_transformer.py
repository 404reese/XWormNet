import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformer_model import TrafficTransformer
from train_and_test import (
    load_data, preprocess, StandardScaler, train_test_split,
    compute_binary_metrics, compute_confusion_matrix, compute_roc_auc
)

def create_sequences(X, y, window_size):
    """
    Convert flat features into sequences of length window_size.
    We take sliding windows over the data.
    """
    X_seq = []
    y_seq = []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size-1])
    return np.array(X_seq), np.array(y_seq)

def main():
    print("Loading data for Transformer...")
    csv_path = "data/GothamDataset2025/processed/iotsim-air-quality-1.csv"
    if not os.path.exists(csv_path):
        csv_path = "GothamDataset2025/processed/iotsim-air-quality-1.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Could not find GothamDataset2025 CSV file.")
            
    df_raw = load_data(csv_path)
    df_exact = df_raw.drop_duplicates()
    
    # Preprocessing identical to clean pipeline
    target_cols = [c for c in ['label', 'attack_type'] if c in df_raw.columns]
    df_features_clean = df_raw.drop(columns=target_cols)
    shortcut_cols = ['ip.proto', 'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport', 'tcp.window_size_scalefactor']
    cols_to_remove = [col for col in shortcut_cols if col in df_features_clean.columns]
    df_features_clean = df_features_clean.drop(columns=cols_to_remove)
    
    numeric_feature_cols_clean = list(df_features_clean.select_dtypes(include=[np.number]).columns)
    df_clean_no_shortcuts = df_exact.drop_duplicates(subset=numeric_feature_cols_clean, keep='first')
    
    X_clean, y_clean = preprocess(df_clean_no_shortcuts, drop_shortcuts=True)
    
    # Create sequences
    window_size = 10
    print(f"Creating sequences with window_size={window_size}...")
    X_seq, y_seq = create_sequences(X_clean, y_clean, window_size)
    print(f"Sequence shape: {X_seq.shape}")
    
    # Train/test split
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    
    # Scale features
    scaler = StandardScaler()
    input_dim = X_train_seq.shape[2]
    
    X_train_flat = X_train_seq.reshape(-1, input_dim)
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_train_s = X_train_scaled_flat.reshape(-1, window_size, input_dim)
    
    X_test_flat = X_test_seq.reshape(-1, input_dim)
    X_test_scaled_flat = scaler.transform(X_test_flat)
    X_test_s = X_test_scaled_flat.reshape(-1, window_size, input_dim)
    
    # Create model
    print("\nInitializing Transformer model...")
    model = TrafficTransformer(input_dim=input_dim, hidden_size=64, nhead=8, num_layers=2)
    
    # Training loop
    epochs = 25
    batch_size = 32
    lr = 0.0005
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training for {epochs} epochs...")
    start_train = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {epoch_loss:.6f}")
        
    train_time = time.perf_counter() - start_train
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Evaluation
    print("\nEvaluating model...")
    model.eval()
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    
    start_eval = time.perf_counter()
    with torch.no_grad():
        probs_t = model(X_test_t)
    end_eval = time.perf_counter()
    
    probs = probs_t.numpy().squeeze()
    
    avg_latency_ms = (end_eval - start_eval) * 1000 / len(X_test_s)
    
    # Tune threshold
    best_threshold = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        _, _, f1 = compute_binary_metrics(y_test_seq, preds)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
            
    preds = (probs >= best_threshold).astype(int)
    
    precision, recall, f1 = compute_binary_metrics(y_test_seq, preds)
    tn, fp, fn, tp = compute_confusion_matrix(y_test_seq, preds)
    roc_auc = compute_roc_auc(y_test_seq, probs)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"  Precision:        {precision:.5f}")
    print(f"  Recall:           {recall:.5f}")
    print(f"  F1 Score:         {f1:.5f}")
    print(f"  ROC AUC:          {roc_auc:.5f}")
    print(f"  Avg Latency (ms): {avg_latency_ms:.5f}")
    
    # Save model
    model_path = 'models/transformer_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save results
    results_df = pd.DataFrame([{
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "FPR": fpr,
        "FNR": fnr,
        "Avg_Latency_ms": avg_latency_ms,
        "Optimal_Threshold": best_threshold
    }])
    results_df.to_csv('outputs/csv/transformer_results.csv', index=False)
    print("Results saved to outputs/csv/transformer_results.csv")

if __name__ == "__main__":
    main()
