import streamlit as st
import pandas as pd
import torch
import os
import joblib
import numpy as np
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
                if not os.path.exists("models/lnn_model.pth"):
                    st.error("LNN model not found. Please train it first.")
                    return
                sd = torch.load("models/lnn_model.pth", weights_only=True)
                input_dim = sd["input_layer.weight"].shape[1]
                
                # Align features
                if sample_features.shape[1] > input_dim:
                    feat_vals = sample_features.values[:, :input_dim]
                elif sample_features.shape[1] < input_dim:
                    feat_vals = np.pad(sample_features.values, ((0,0), (0, input_dim - sample_features.shape[1])))
                else:
                    feat_vals = sample_features.values

                model = LNN(input_dim=input_dim, hidden_dim=16, num_steps=6, dt=0.1)
                model.load_state_dict(sd)
                model.eval()
                with torch.no_grad():
                    X_t = torch.tensor(feat_vals, dtype=torch.float32)
                    outputs = model(X_t)
                    probs = torch.sigmoid(outputs)
                    prediction = (probs > 0.5).int().numpy()[0]
            else:
                if not os.path.exists("models/rf_model.pkl"):
                    st.error("RF model not found. Please train it first.")
                    return
                model = joblib.load("models/rf_model.pkl")
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

    # LNN LIQUID PARAMETERS TAB
    st.divider()
    st.header("💧 LNN Liquid Parameters Analysis")
    
    if st.button("Generate Liquid Parameter Visualizations"):
        if 'sample_features' not in locals():
            st.error("Please load or upload a sample first.")
            return
            
        with st.spinner("Analyzing liquid dynamics..."):
            from visualize_liquid_params import plot_liquid_gates_over_time, plot_hidden_evolution_heatmap, compare_lnn_vs_lstm
            from lstm_model import LSTMClassifier
            
            # Load LNN model
            if not os.path.exists("models/lnn_model.pth"):
                st.error("LNN model not found. Please train it first.")
                return
                
            sd_lnn = torch.load("models/lnn_model.pth", weights_only=True)
            input_dim = sd_lnn["input_layer.weight"].shape[1]
            
            lnn_model = LNN(input_dim=input_dim, hidden_dim=16, num_steps=6, dt=0.1)
            lnn_model.load_state_dict(sd_lnn)
            lnn_model.eval()
            
            # Load LSTM model if exists, else pass None
            lstm_model = None
            if os.path.exists("models/lstm_model.pth"):
                sd_lstm = torch.load("models/lstm_model.pth", weights_only=True)
                lstm_input_dim = sd_lstm["lstm.weight_ih_l0"].shape[1]
                lstm_model = LSTMClassifier(input_dim=lstm_input_dim, hidden_size=64)
                lstm_model.load_state_dict(sd_lstm)
                lstm_model.eval()
                
            # Prepare data sequence for plotting (batch_size=1)
            import numpy as np
            if sample_features.shape[1] > input_dim:
                feat_vals = sample_features.values[:, :input_dim]
            elif sample_features.shape[1] < input_dim:
                feat_vals = np.pad(sample_features.values, ((0,0), (0, input_dim - sample_features.shape[1])))
            else:
                feat_vals = sample_features.values
                
            X_seq = torch.tensor(feat_vals, dtype=torch.float32)
            
            # Generate plots
            plot_liquid_gates_over_time(lnn_model, X_seq)
            plot_hidden_evolution_heatmap(lnn_model, X_seq)
            compare_lnn_vs_lstm(lnn_model, lstm_model, X_seq)
            
            st.success("Visualizations generated successfully!")
            
            tab1, tab2, tab3 = st.tabs(["Gate Evolution", "Hidden State", "LNN vs LSTM"])
            
            with tab1:
                st.image("figures/lnn_liquid_parameters.png", caption="LNN Gate Evolution Over Time")
                st.info("The decay and input gates dynamically adapt at each ODE unrolling step based on the hidden state.")
                
            with tab2:
                st.image("figures/lnn_hidden_heatmap.png", caption="LNN Hidden State Evolution Heatmap")
                st.info("Visualizes how individual hidden units evolve continuously during processing.")
                
            with tab3:
                st.image("figures/lnn_vs_lstm_parameters.png", caption="LNN vs LSTM Parameter Comparison")
                st.info("LNN's liquid parameters change over time → enables adaptive behavior for zero-day worms (unlike LSTM's fixed weights)")

