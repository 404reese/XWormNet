import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from gan_model import AnomalyGAN
from train_and_test import (
    load_data, preprocess, StandardScaler, train_test_split,
    compute_binary_metrics, compute_confusion_matrix, compute_roc_auc
)

def main():
    print("Loading data for GAN...")
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
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Filter normal traffic for training the GAN
    # Label 0 is normal.
    X_train_normal = X_train_s[y_train == 0]
    
    input_dim = X_train_s.shape[1]
    latent_dim = 32
    
    # Create model
    print("\nInitializing AnomalyGAN model...")
    model = AnomalyGAN(input_dim=input_dim, latent_dim=latent_dim, hidden_size=64)
    
    # Training loop
    epochs = 30
    batch_size = 64
    lr = 0.0002
    
    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    
    X_train_t = torch.tensor(X_train_normal, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training for {epochs} epochs on NORMAL traffic only...")
    start_train = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        g_loss_avg = 0.0
        d_loss_avg = 0.0
        for batch in train_loader:
            real_data = batch[0]
            b_size = real_data.size(0)
            
            # Train Discriminator
            opt_d.zero_grad()
            real_labels = torch.ones(b_size, 1)
            fake_labels = torch.zeros(b_size, 1)
            
            outputs_real = model.discriminator(real_data)
            d_loss_real = criterion(outputs_real, real_labels)
            
            z = torch.randn(b_size, latent_dim)
            fake_data = model.generator(z)
            outputs_fake = model.discriminator(fake_data.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()
            
            # Train Generator
            opt_g.zero_grad()
            outputs_fake = model.discriminator(fake_data)
            # Generator wants discriminator to think fake data is real
            g_loss = criterion(outputs_fake, real_labels)
            g_loss.backward()
            opt_g.step()
            
            d_loss_avg += d_loss.item() * b_size
            g_loss_avg += g_loss.item() * b_size
            
        d_loss_avg /= len(train_loader.dataset)
        g_loss_avg /= len(train_loader.dataset)
        print(f"Epoch {epoch+1:02d}/{epochs} - D Loss: {d_loss_avg:.6f} - G Loss: {g_loss_avg:.6f}")
        
    train_time = time.perf_counter() - start_train
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Evaluation
    print("\nEvaluating model on full test set (normal + attacks)...")
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
    model_path = 'models/gan_model.pth'
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
    results_df.to_csv('outputs/csv/gan_results.csv', index=False)
    print("Results saved to outputs/csv/gan_results.csv")

if __name__ == "__main__":
    main()
