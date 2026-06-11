import streamlit as st
import pandas as pd
import torch
import os
import joblib
from lnn_model import LNN
from explainability import explain_with_shap, explain_with_lime

def render():
    st.header("🔍 Explainability (SHAP/LIME)")

    # UPLOAD TEST SAMPLE (REAL)
    uploaded_file = st.file_uploader("Upload Test Sample (CSV)", type=["csv"])
    if uploaded_file:
        sample = pd.read_csv(uploaded_file).iloc[:1]
    else:
        # Load first row from test data
        default_path = "data/GothamDataset2025/processed/iotsim-air-quality-1.csv"
        if os.path.exists(default_path):
            sample = pd.read_csv(default_path, nrows=1)
        else:
            st.error(f"Default dataset not found at {default_path}. Please upload a CSV.")
            return

    # Drop target columns for prediction
    bad_substrings = ['ip', 'port', 'proto', 'mac', 'flow_id']
    sample_features = sample.drop(columns=["label", "attack_type", "source_file"], errors="ignore").select_dtypes(include=["number"])
    cols_to_remove = [col for col in sample_features.columns if any(sub in col.lower() for sub in bad_substrings)]
    sample_features = sample_features.drop(columns=cols_to_remove).fillna(0.0)

    # SELECT MODEL (REAL)
    model_type = st.selectbox("Select Model", ["LNN", "RF"])

    # PREDICT (REAL)
    if st.button("🔮 Predict & Explain"):
        with st.spinner("Running inference and generating explanations..."):
            # Load model
            if model_type == "LNN":
                if not os.path.exists("lnn_model.pth"):
                    st.error("LNN model not found. Please train it first.")
                    return
                model = LNN(input_dim=sample_features.shape[1], hidden_dim=16, num_steps=6, dt=0.1)
                model.load_state_dict(torch.load("lnn_model.pth", weights_only=True))
                model.eval()
                with torch.no_grad():
                    X_t = torch.tensor(sample_features.values, dtype=torch.float32)
                    outputs = model(X_t)
                    probs = torch.sigmoid(outputs)
                    prediction = (probs > 0.5).int().numpy()[0]
            else:
                if not os.path.exists("rf_model.pkl"):
                    st.error("RF model not found. Please train it first.")
                    return
                model = joblib.load("rf_model.pkl")
                prediction = model.predict(sample_features)[0]
            
            # SHOW PREDICTION (REAL)
            if prediction == 0:
                st.success("🟢 Prediction: Normal (label=0)")
            else:
                st.error("🔴 Prediction: Attack (label=1)")
            
            # EXPLANATIONS (REAL)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("SHAP Explanation")
                shap_result = explain_with_shap(sample_features, model, model_type)
                st.bar_chart(shap_result.sort_values(ascending=False).head(10))
            
            with col2:
                st.subheader("LIME Explanation")
                lime_result = explain_with_lime(sample_features, model, model_type)
                st.bar_chart(lime_result.sort_values(ascending=False).head(10))
            
            # TOP FEATURES (REAL)
            st.subheader("Top 5 Features Contributing to Prediction (SHAP)")
            top_features = shap_result.nlargest(5)
            for feature, value in top_features.items():
                st.write(f"• **{feature}**: {value:.3f}")
