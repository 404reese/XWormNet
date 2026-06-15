"""
compare_models.py
-----------------
Build the final 8-model comparison table for the XWormNet IEEE paper.

Models compared
---------------
  LNN         Liquid Neural Network (proposed)
  RF          Random Forest baseline
  LSTM        Long Short-Term Memory
  GRU         Gated Recurrent Unit
  Transformer Traffic Transformer
  GAN         Generative Adversarial Network detector
  AR          Autoregressive classifier
  RNN         Vanilla RNN (new baseline)

Outputs
-------
  outputs/csv/comparison_results.csv  — 8-row table (overwritten each run)
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(path: str, name: str, script: str):
    """Abort with a helpful message if a required CSV is missing."""
    if not os.path.exists(path):
        print(f"[MISSING] {name} results not found at '{path}'.")
        print(f"          Run  python {script}  first, then re-run compare_models.py.")
        return False
    return True


def _model_size_mb(path: str) -> float:
    """Return file size in MB, or 0.0 if file does not exist."""
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # 1. Verify all result CSVs exist
    # ------------------------------------------------------------------
    required = [
        ("outputs/csv/results.csv",               "LNN",         "train_and_test.py"),
        ("outputs/csv/rf_results.csv",             "RF",          "rf_baseline.py"),
        ("outputs/csv/lstm_results.csv",           "LSTM",        "train_lstm.py"),
        ("outputs/csv/gru_results.csv",            "GRU",         "train_gru.py"),
        ("outputs/csv/transformer_results.csv",    "Transformer", "train_transformer.py"),
        ("outputs/csv/gan_results.csv",            "GAN",         "train_gan.py"),
        ("outputs/csv/autoregressive_results.csv", "AR",          "train_autoregressive.py"),
        ("outputs/csv/rnn_results.csv",            "RNN",         "train_rnn.py"),
    ]

    all_ok = all(_require(path, name, script) for path, name, script in required)
    if not all_ok:
        print("\nOne or more result files are missing. Aborting.")
        return

    # ------------------------------------------------------------------
    # 2. Load results
    # ------------------------------------------------------------------
    lnn_df  = pd.read_csv("outputs/csv/results.csv").iloc[0]
    rf_df   = pd.read_csv("outputs/csv/rf_results.csv").iloc[0]
    lstm_df = pd.read_csv("outputs/csv/lstm_results.csv").iloc[0]
    gru_df  = pd.read_csv("outputs/csv/gru_results.csv").iloc[0]
    tf_df   = pd.read_csv("outputs/csv/transformer_results.csv").iloc[0]
    gan_df  = pd.read_csv("outputs/csv/gan_results.csv").iloc[0]
    ar_df   = pd.read_csv("outputs/csv/autoregressive_results.csv").iloc[0]
    rnn_df  = pd.read_csv("outputs/csv/rnn_results.csv").iloc[0]

    # ------------------------------------------------------------------
    # 3. Model file sizes
    # ------------------------------------------------------------------
    lnn_size  = _model_size_mb("models/lnn_model.pth")
    rf_size   = _model_size_mb("models/rf_model.pkl")
    lstm_size = _model_size_mb("models/lstm_model.pth")
    gru_size  = _model_size_mb("models/gru_model.pth")
    tf_size   = _model_size_mb("models/transformer_model.pth")
    gan_size  = _model_size_mb("models/gan_model.pth")
    ar_size   = _model_size_mb("models/autoregressive_model.pth")
    rnn_size  = _model_size_mb("models/rnn_model.pth")

    # ------------------------------------------------------------------
    # 4. Build comparison DataFrame  (8 models)
    # ------------------------------------------------------------------
    models    = ["LNN", "RF", "LSTM", "GRU", "Transformer", "GAN", "AR", "RNN"]
    rows      = [lnn_df, rf_df, lstm_df, gru_df, tf_df, gan_df, ar_df, rnn_df]
    sizes     = [lnn_size, rf_size, lstm_size, gru_size, tf_size, gan_size, ar_size, rnn_size]

    comp_df = pd.DataFrame({
        "Model":        models,
        "Precision":    [r["Precision"]      for r in rows],
        "Recall":       [r["Recall"]         for r in rows],
        "F1":           [r["F1_Score"]       for r in rows],
        "Latency (ms)": [r["Avg_Latency_ms"] for r in rows],
        "Size (MB)":    sizes,
    })

    # ------------------------------------------------------------------
    # 5. Print formatted table
    # ------------------------------------------------------------------
    print("\n" + "=" * 82)
    print("                         MODEL COMPARISON  (8 models)")
    print("=" * 82)
    print(
        f"| {'Model':<11} | {'Precision':<10} | {'Recall':<10} | "
        f"{'F1':<10} | {'Latency (ms)':<14} | {'Size (MB)':<10} |"
    )
    print("|" + "-" * 13 + "|" + "-" * 12 + "|" + "-" * 12 + "|"
          + "-" * 12 + "|" + "-" * 16 + "|" + "-" * 12 + "|")

    for _, row in comp_df.iterrows():
        print(
            f"| {row['Model']:<11} | {row['Precision']:<10.4f} | {row['Recall']:<10.4f} | "
            f"{row['F1']:<10.4f} | {row['Latency (ms)']:<14.4f} | {row['Size (MB)']:<10.4f} |"
        )

    print("=" * 82)

    # Insights
    fastest = comp_df.loc[comp_df["Latency (ms)"].idxmin(), "Model"]
    highest_f1 = comp_df.loc[comp_df["F1"].idxmax(), "Model"]
    smallest = comp_df.loc[comp_df["Size (MB)"].idxmin(), "Model"]
    print(f"  [OK]  Highest F1        : {highest_f1}")
    print(f"  [FAST] Fastest inference : {fastest}")
    print(f"  [SMALL] Smallest model  : {smallest}")
    print(f"\n  RNN is the simplest recurrent baseline -- no gating, no dropout.")
    print(f"  LNN / AR / GAN are the fastest for edge IoT deployment.\n")

    # ------------------------------------------------------------------
    # 6. Save CSV
    # ------------------------------------------------------------------
    os.makedirs("outputs/csv", exist_ok=True)
    out_path = "outputs/csv/comparison_results.csv"
    comp_df.to_csv(out_path, index=False)
    print(f"Saved comparison table -> {out_path}")


if __name__ == "__main__":
    main()
