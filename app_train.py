import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import time
from lnn_model import LNN
from lstm_model import LSTMClassifier
from gru_model import GRUClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

def inference_latency(model, X_test, model_type):
    start = time.time()
    if model_type == "LNN":
        X_t = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            model(X_t)
    elif model_type in ["LSTM", "GRU"]:
        X_t = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            model(X_t)
    else:
        model.predict(X_test)
    end = time.time()
    return ((end - start) / len(X_test)) * 1000

def render():
    st.header("🎯 Train New Model")

    # SELECT MODEL (REAL)
    model_type = st.selectbox("Select Model", ["LNN", "RF", "LSTM", "GRU"])

    # UPLOAD DATASET (REAL)
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        # Default: load GothamDataset2025 CSV
        default_path = "data/GothamDataset2025/processed/iotsim-air-quality-1.csv"
        if os.path.exists(default_path):
            data = pd.read_csv(default_path, nrows=10000) # limit to 10k for fast dashboard training
            st.info(f"Loaded default dataset: {default_path} (10,000 rows)")
        else:
            st.error(f"Default dataset not found at {default_path}. Please upload a CSV.")
            return

    # PREPARE DATA (REAL)
    # Add label if missing
    if "label" not in data.columns:
        if "attack_type" in data.columns:
            data["label"] = (data["attack_type"].astype(str).str.lower() != "benign").astype(int)
        else:
            data["label"] = 0  # All benign

    # Drop non-numeric
    X = data.drop(columns=["label", "attack_type", "source_file"], errors="ignore")
    X = X.select_dtypes(include=["number"])
    X = X.dropna(axis=1)
    # Normalize labels to integer 0/1 for training. Handle numeric and string labels.
    y_raw = data["label"]
    # Try converting to numeric first (handles 0/1 already stored as numbers or numeric strings)
    y_numeric = pd.to_numeric(y_raw, errors="coerce")
    if y_numeric.isna().any():
        # For non-numeric labels (e.g., 'benign' / 'malicious' or other strings), map 'benign'->0 else 1
        y_str = y_raw.astype(str).str.lower()
        y_mapped = (y_str != "benign").astype(int)
        y = y_numeric.fillna(y_mapped).astype(int)
    else:
        y = y_numeric.astype(int)

    if len(X) == 0:
        st.error("No numeric data found to train on.")
        return

    # TRAIN/TEST SPLIT (REAL)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y if len(y.unique()) > 1 else None)

    # HYPERPARAMETERS
    epochs = st.slider("Epochs", 1, 100, 5) if model_type == "LNN" else None
    hidden_size = st.slider("Hidden Size", 16, 256, 16) if model_type == "LNN" else None

    # TRAIN BUTTON (REAL)
    if st.button("🚀 Train Model"):
        with st.spinner(f"Training {model_type}..."):
            if model_type == "LNN":
                # Train LNN (REAL)
                # LNN usually requires input_dim, hidden_dim, num_steps
                model = LNN(input_dim=X_train.shape[1], hidden_dim=hidden_size, num_steps=6, dt=0.1)
                criterion = torch.nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                losses = []
                X_tr_t = torch.tensor(X_train.values, dtype=torch.float32)
                y_tr_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = model(X_tr_t)
                    loss = criterion(outputs, y_tr_t)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                
                # Save model (REAL)
                torch.save(model.state_dict(), "models/lnn_model.pth")
                
            elif model_type in ["LSTM", "GRU"]:
                window_size = 10
                X_vals = X_train.values
                y_vals = y_train.values
                
                X_seq, y_seq = [], []
                for i in range(len(X_vals) - window_size + 1):
                    X_seq.append(X_vals[i:i+window_size])
                    y_seq.append(y_vals[i+window_size-1])
                X_seq = np.array(X_seq)
                y_seq = np.array(y_seq)
                
                if model_type == "LSTM":
                    model = LSTMClassifier(input_dim=X_train.shape[1], hidden_size=64)
                else:
                    model = GRUClassifier(input_dim=X_train.shape[1], hidden_size=64)
                    
                criterion = torch.nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                X_tr_t = torch.tensor(X_seq, dtype=torch.float32)
                y_tr_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)
                
                train_dataset = TensorDataset(X_tr_t, y_tr_t)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                
                losses = []
                for epoch in range(20):
                    epoch_loss = 0.0
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item() * batch_x.size(0)
                    losses.append(epoch_loss / len(train_dataset))
                    
                if model_type == "LSTM":
                    torch.save(model.state_dict(), "models/lstm_model.pth")
                else:
                    torch.save(model.state_dict(), "models/gru_model.pth")
                
            else:
                # Train RF (REAL)
                model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
                model.fit(X_train, y_train)
                
                # Save model (REAL)
                import joblib
                joblib.dump(model, "models/rf_model.pkl")
            
        # EVALUATE (REAL)
        with st.spinner("Evaluating..."):
            if model_type == "LNN":
                X_te_t = torch.tensor(X_test.values, dtype=torch.float32)
                with torch.no_grad():
                    outputs = model(X_te_t)
                    probs = torch.sigmoid(outputs)
                    y_pred = (probs > 0.5).int().numpy().flatten()
            elif model_type in ["LSTM", "GRU"]:
                window_size = 10
                X_vals = X_test.values
                y_vals = y_test.values
                
                X_seq_test, y_seq_test = [], []
                for i in range(len(X_vals) - window_size + 1):
                    X_seq_test.append(X_vals[i:i+window_size])
                    y_seq_test.append(y_vals[i+window_size-1])
                X_seq_test = np.array(X_seq_test)
                y_test_adjusted = np.array(y_seq_test)
                
                X_te_t = torch.tensor(X_seq_test, dtype=torch.float32)
                with torch.no_grad():
                    probs = model(X_te_t)
                    y_pred = (probs > 0.5).int().numpy().flatten()
                y_test = y_test_adjusted
            else:
                y_pred = model.predict(X_test)
            
            # Avoid division by zero warnings
            if len(y_test.unique()) > 1:
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
            else:
                precision = recall = f1 = 0.0

            if model_type in ["LSTM", "GRU"]:
                # For inference latency on sequences we need sequential data
                latency = inference_latency(model, X_seq_test[:100], model_type)
            else:
                latency = inference_latency(model, X_test.values[:100], model_type)
            
            # Approximate size
            if model_type == "LNN":
                size = os.path.getsize("models/lnn_model.pth") / (1024 * 1024) if os.path.exists("models/lnn_model.pth") else 0.007
            elif model_type == "LSTM":
                size = os.path.getsize("models/lstm_model.pth") / (1024 * 1024) if os.path.exists("models/lstm_model.pth") else 0.0
            elif model_type == "GRU":
                size = os.path.getsize("models/gru_model.pth") / (1024 * 1024) if os.path.exists("models/gru_model.pth") else 0.0
            else:
                size = os.path.getsize("models/rf_model.pkl") / (1024 * 1024) if os.path.exists("models/rf_model.pkl") else 0.232
            
        # SHOW RESULTS (REAL)
        st.success("✅ Training Complete!")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Precision", f"{precision:.3f}")
        col2.metric("Recall", f"{recall:.3f}")
        col3.metric("F1 Score", f"{f1:.3f}")
        col4.metric("Latency", f"{latency:.3f} ms")
        col5.metric("Model Size", f"{size:.3f} MB")
        
        # TRAINING LOSS CURVE (REAL)
        if model_type in ["LNN", "LSTM", "GRU"]:
            st.subheader("Training Loss")
            st.line_chart(losses)
        
        # UPDATE COMPARISON (REAL)
        comp_path = "outputs/csv/comparison_results.csv"
        if os.path.exists(comp_path):
            comparison = pd.read_csv(comp_path)
        else:
            comparison = pd.DataFrame(columns=["Model", "Precision", "Recall", "F1", "Latency (ms)", "Size (MB)"])
            
        new_row = pd.DataFrame({
            "Model": [model_type],
            "Precision": [precision],
            "Recall": [recall],
            "F1": [f1],
            "Latency (ms)": [latency],
            "Size (MB)": [size]
        })
        
        if model_type in comparison["Model"].values:
            comparison = comparison[comparison["Model"] != model_type]
        comparison = pd.concat([comparison, new_row], ignore_index=True)
        comparison.to_csv(comp_path, index=False)
        st.info("📊 Results saved to outputs/csv/comparison_results.csv")
