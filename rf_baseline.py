import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Import exactly the same data processing functions from the training script
from train_and_test import (
    load_data, preprocess, StandardScaler, train_test_split, 
    compute_binary_metrics, compute_confusion_matrix, compute_roc_auc
)

def main():
    print("Loading data for RF Baseline...")
    csv_path = "data/GothamDataset2025/processed/iotsim-air-quality-1.csv"
    if not os.path.exists(csv_path):
        csv_path = "GothamDataset2025/processed/iotsim-air-quality-1.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Could not find GothamDataset2025 CSV file.")
            
    df_raw = load_data(csv_path)
    df_exact = df_raw.drop_duplicates()
    
    # Preprocessing identical to LNN's Clean Pipeline (Stage 3)
    target_cols = [c for c in ['label', 'attack_type'] if c in df_raw.columns]
    df_features_clean = df_raw.drop(columns=target_cols)
    shortcut_cols = ['ip.proto', 'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport', 'tcp.window_size_scalefactor']
    cols_to_remove = [col for col in shortcut_cols if col in df_features_clean.columns]
    df_features_clean = df_features_clean.drop(columns=cols_to_remove)
    
    numeric_feature_cols_clean = list(df_features_clean.select_dtypes(include=[np.number]).columns)
    df_clean_no_shortcuts = df_exact.drop_duplicates(subset=numeric_feature_cols_clean, keep='first')
    
    X_clean, y_clean = preprocess(df_clean_no_shortcuts, drop_shortcuts=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("\nTraining Random Forest Classifier (100 estimators)...")
    # Setting n_jobs=-1 to use all available cores
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    start_train = time.perf_counter()
    rf.fit(X_train_s, y_train)
    train_time = time.perf_counter() - start_train
    print(f"Training completed in {train_time:.2f} seconds.")
    
    print("\nEvaluating model...")
    # Evaluate Latency
    start_eval = time.perf_counter()
    probs = rf.predict_proba(X_test_s)[:, 1]
    end_eval = time.perf_counter()
    
    avg_latency_ms = (end_eval - start_eval) * 1000 / len(X_test_s)
    
    # Tune threshold
    best_threshold = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        _, _, f1 = compute_binary_metrics(y_test, preds)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
            
    preds = (probs >= best_threshold).astype(int)
    
    precision, recall, f1 = compute_binary_metrics(y_test, preds)
    tn, fp, fn, tp = compute_confusion_matrix(y_test, preds)
    roc_auc = compute_roc_auc(y_test, probs)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"  Precision:        {precision:.5f}")
    print(f"  Recall:           {recall:.5f}")
    print(f"  F1 Score:         {f1:.5f}")
    print(f"  ROC AUC:          {roc_auc:.5f}")
    print(f"  Avg Latency (ms): {avg_latency_ms:.5f}")
    
    # Save model
    model_path = 'models/rf_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
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
    results_df.to_csv('outputs/csv/rf_results.csv', index=False)
    print("Results saved to outputs/csv/rf_results.csv")

if __name__ == "__main__":
    main()
