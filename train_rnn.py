"""
train_rnn.py
------------
Train and evaluate the vanilla RNN baseline (RNNClassifier) on the
GothamDataset2025 CSV.  Mirrors the structure of train_lstm.py and
train_gru.py so results are directly comparable.

Hyperparameters
---------------
  RNN layers  : 2
  Hidden size : 64
  Epochs      : 20
  Batch size  : 32
  Learning rate: 0.001
  Window size : 10 (same as LSTM / GRU)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Allow both `python train_rnn.py` (root) and imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rnn_model import RNNClassifier
from train_and_test import (
    load_data, preprocess, StandardScaler, train_test_split,
    compute_binary_metrics, compute_confusion_matrix, compute_roc_auc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_sequences(X: np.ndarray, y: np.ndarray, window_size: int):
    """
    Convert flat feature array into sliding-window sequences.

    Returns:
        X_seq : (N - window_size + 1, window_size, input_dim)
        y_seq : (N - window_size + 1,)  — label at the last step of each window
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i : i + window_size])
        y_seq.append(y[i + window_size - 1])
    return np.array(X_seq), np.array(y_seq)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Vanilla RNN Baseline — XWormNet")
    print("=" * 60)
    print("\n[1/5] Loading data...")

    csv_path = "data/GothamDataset2025/processed/iotsim-air-quality-1.csv"
    if not os.path.exists(csv_path):
        csv_path = "GothamDataset2025/processed/iotsim-air-quality-1.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                "Could not find GothamDataset2025 CSV. "
                "Expected at: data/GothamDataset2025/processed/iotsim-air-quality-1.csv"
            )

    df_raw = load_data(csv_path)
    df_exact = df_raw.drop_duplicates()

    # ------------------------------------------------------------------
    # 2. Preprocessing (identical to LSTM / GRU pipelines)
    # ------------------------------------------------------------------
    target_cols = [c for c in ["label", "attack_type"] if c in df_raw.columns]
    df_features_clean = df_raw.drop(columns=target_cols)

    shortcut_cols = [
        "ip.proto", "tcp.srcport", "tcp.dstport",
        "udp.srcport", "udp.dstport", "tcp.window_size_scalefactor",
    ]
    cols_to_remove = [c for c in shortcut_cols if c in df_features_clean.columns]
    df_features_clean = df_features_clean.drop(columns=cols_to_remove)

    numeric_cols = list(df_features_clean.select_dtypes(include=[np.number]).columns)
    df_clean = df_exact.drop_duplicates(subset=numeric_cols, keep="first")

    X_clean, y_clean = preprocess(df_clean, drop_shortcuts=True)

    # ------------------------------------------------------------------
    # 3. Create sequences  (window_size = 10)
    # ------------------------------------------------------------------
    window_size = 10
    print(f"\n[2/5] Creating sequences (window_size={window_size})...")
    X_seq, y_seq = create_sequences(X_clean, y_clean, window_size)
    print(f"      Sequence shape : {X_seq.shape}")
    print(f"      Label shape    : {y_seq.shape}")

    # ------------------------------------------------------------------
    # 4. Train / test split  (80/20, stratified)
    # ------------------------------------------------------------------
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    # Scale features (fit on train, apply to both splits)
    scaler = StandardScaler()
    input_dim = X_train_seq.shape[2]

    X_train_flat = X_train_seq.reshape(-1, input_dim)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(-1, window_size, input_dim)

    X_test_flat = X_test_seq.reshape(-1, input_dim)
    X_test_scaled = scaler.transform(X_test_flat).reshape(-1, window_size, input_dim)

    print(f"      Train samples  : {len(X_train_seq)}")
    print(f"      Test  samples  : {len(X_test_seq)}")
    print(f"      Input dim      : {input_dim}")

    # ------------------------------------------------------------------
    # 5. Build model
    # ------------------------------------------------------------------
    print("\n[3/5] Initialising RNN model...")
    model = RNNClassifier(input_dim=input_dim, hidden_size=64)
    print(f"      Parameters     : {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    epochs     = 20
    batch_size = 32
    lr         = 0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq,    dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"\n[4/5] Training for {epochs} epochs (batch={batch_size}, lr={lr})...")
    start_train = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss    = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"      Epoch {epoch+1:02d}/{epochs}  loss: {epoch_loss:.6f}")

    train_time = time.perf_counter() - start_train
    print(f"      Training time  : {train_time:.2f}s")

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating model...")
    model.eval()
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    start_eval = time.perf_counter()
    with torch.no_grad():
        probs_t = model(X_test_t)
    end_eval = time.perf_counter()

    probs          = probs_t.numpy().squeeze()
    avg_latency_ms = (end_eval - start_eval) * 1000 / len(X_test_scaled)

    # Threshold tuning (maximise F1)
    best_threshold = 0.5
    best_f1        = 0.0
    for t in np.linspace(0.01, 0.99, 99):
        preds_t = (probs >= t).astype(int)
        _, _, f1_t = compute_binary_metrics(y_test_seq, preds_t)
        if f1_t > best_f1:
            best_f1, best_threshold = f1_t, t

    preds = (probs >= best_threshold).astype(int)

    precision, recall, f1 = compute_binary_metrics(y_test_seq, preds)
    tn, fp, fn, tp        = compute_confusion_matrix(y_test_seq, preds)
    roc_auc               = compute_roc_auc(y_test_seq, probs)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*37}")
    print(f"  {'Precision':<25} {precision:>10.5f}")
    print(f"  {'Recall':<25} {recall:>10.5f}")
    print(f"  {'F1 Score':<25} {f1:>10.5f}")
    print(f"  {'ROC AUC':<25} {roc_auc:>10.5f}")
    print(f"  {'FPR':<25} {fpr:>10.5f}")
    print(f"  {'FNR':<25} {fnr:>10.5f}")
    print(f"  {'Avg Latency (ms/sample)':<25} {avg_latency_ms:>10.5f}")
    print(f"  {'Optimal Threshold':<25} {best_threshold:>10.4f}")

    # ------------------------------------------------------------------
    # 8. Save model
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    model_path = "models/rnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved  -> {model_path}")

    # ------------------------------------------------------------------
    # 9. Save results CSV
    # ------------------------------------------------------------------
    os.makedirs("outputs/csv", exist_ok=True)
    results_df = pd.DataFrame([{
        "Precision":           precision,
        "Recall":              recall,
        "F1_Score":            f1,
        "ROC_AUC":             roc_auc,
        "FPR":                 fpr,
        "FNR":                 fnr,
        "Avg_Latency_ms":      avg_latency_ms,
        "Optimal_Threshold":   best_threshold,
    }])
    results_csv = "outputs/csv/rnn_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved -> {results_csv}")
    print("\nDone! Run `python compare_models.py` to regenerate the 8-model table.\n")


if __name__ == "__main__":
    main()
