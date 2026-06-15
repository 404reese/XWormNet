"""
app_explainability.py
---------------------
Streamlit page: Explainability (SHAP / LIME / T-SHAP).

Supports 8 model types:  LNN, RF, LSTM, GRU, RNN, Transformer, AR, GAN
Uses real shap + lime under the hood via src/explainability.py.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Make sure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.explainability import explain_with_shap, explain_with_lime, explain_with_tshap
from src.lnn_model       import LNN
from src.lstm_model      import LSTMClassifier
from src.gru_model       import GRUClassifier
from src.rnn_model       import RNNClassifier
from src.transformer_model import TrafficTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CSV = "data/GothamDataset2025/processed/iotsim-air-quality-1.csv"
_BAD_SUBS    = ["ip", "port", "proto", "mac", "flow_id"]
_DROP_COLS   = ["label", "attack_type", "source_file"]
_WINDOW_SIZE = 10
_SEQ_MODELS  = {"LSTM", "GRU", "RNN", "Transformer", "AR"}
_ALL_MODELS  = ["LNN", "RF", "LSTM", "GRU", "RNN", "Transformer", "GAN", "AR"]

_MODEL_PATHS = {
    "LNN":         "models/lnn_model.pth",
    "RF":          "models/rf_model.pkl",
    "LSTM":        "models/lstm_model.pth",
    "GRU":         "models/gru_model.pth",
    "RNN":         "models/rnn_model.pth",
    "Transformer": "models/transformer_model.pth",
    "GAN":         "models/gan_model.pth",
    "AR":          "models/autoregressive_model.pth",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Drop target / identifier columns and keep only clean numeric features."""
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns], errors="ignore")
    df = df.select_dtypes(include=[np.number])
    bad = [c for c in df.columns if any(s in c.lower() for s in _BAD_SUBS)]
    return df.drop(columns=bad).fillna(0.0)


def _load_model(model_type: str):
    """Load and return a trained model.  Returns None on failure."""
    path = _MODEL_PATHS.get(model_type, "")
    if not os.path.exists(path):
        st.error(
            f"Model file not found: `{path}`.  "
            f"Please train **{model_type}** first (Training tab)."
        )
        return None

    try:
        if model_type == "RF":
            return joblib.load(path)

        sd = torch.load(path, weights_only=True)

        if model_type == "LNN":
            input_dim = sd["input_layer.weight"].shape[1]
            m = LNN(input_dim=input_dim, hidden_dim=16, num_steps=6, dt=0.1)

        elif model_type == "LSTM":
            input_dim = sd["lstm.weight_ih_l0"].shape[1]
            m = LSTMClassifier(input_dim=input_dim, hidden_size=64)

        elif model_type == "GRU":
            input_dim = sd["gru.weight_ih_l0"].shape[1]
            m = GRUClassifier(input_dim=input_dim, hidden_size=64)

        elif model_type == "RNN":
            input_dim = sd["rnn.weight_ih_l0"].shape[1]
            m = RNNClassifier(input_dim=input_dim, hidden_size=64)

        elif model_type == "Transformer":
            input_dim = sd["embedding.weight"].shape[1]
            m = TrafficTransformer(input_dim=input_dim, hidden_size=64, nhead=8, num_layers=2)

        else:
            # GAN discriminator / AR model — try to load via sys.path models
            try:
                from src.autoregressive_model import AutoregressiveClassifier
                input_dim = next(iter(sd.values())).shape[-1]
                m = AutoregressiveClassifier(window_size=_WINDOW_SIZE,
                                             input_dim=input_dim, ar_order=5)
            except Exception:
                st.warning(f"Cannot reconstruct {model_type} architecture for explainability. Using RF path.")
                return None

        m.load_state_dict(sd)
        m.eval()
        return m

    except Exception as exc:
        st.error(f"Failed to load {model_type}: {exc}")
        return None


def _align_features(sample: pd.DataFrame, n_needed: int) -> pd.DataFrame:
    """Trim or zero-pad columns of `sample` to match `n_needed`."""
    n_have = sample.shape[1]
    if n_have == n_needed:
        return sample
    if n_have > n_needed:
        return sample.iloc[:, :n_needed]
    # pad
    pad_cols = {f"_pad_{i}": [0.0] for i in range(n_needed - n_have)}
    return pd.concat([sample, pd.DataFrame(pad_cols)], axis=1)


def _predict(model, sample: pd.DataFrame, model_type: str) -> int:
    """Run inference and return 0 (normal) or 1 (attack)."""
    X = sample.values.astype(np.float32)
    if model_type == "RF":
        return int(model.predict(X)[0])
    model.eval()
    with torch.no_grad():
        if model_type in _SEQ_MODELS:
            # tile the single row into a window
            seq = np.tile(X, (_WINDOW_SIZE, 1)).reshape(1, _WINDOW_SIZE, -1)
            t   = torch.tensor(seq, dtype=torch.float32)
        else:
            t   = torch.tensor(X, dtype=torch.float32)
        out  = model(t)
        prob = torch.sigmoid(out).item() if model_type == "LNN" else out.item()
    return int(prob > 0.5)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _bar_chart(series: pd.Series, title: str, color: str = "#4C72B0", top_k: int = 10):
    """Render a horizontal bar chart of SHAP / LIME values."""
    s = series.dropna()
    s = s.reindex(s.abs().nlargest(top_k).index)   # top-k by magnitude

    if s.empty or s.abs().sum() == 0:
        st.caption(f"{title}: all values are zero — try a different model or sample.")
        return

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(s) * 0.4)))
    colors  = [color if v >= 0 else "#DD8452" for v in s.values]
    ax.barh(s.index[::-1], s.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _heatmap_chart(df: pd.DataFrame, title: str):
    """Render a timestep × feature heatmap for T-SHAP."""
    if df.empty or df.abs().values.sum() == 0:
        st.caption(f"{title}: all T-SHAP values are zero.")
        return

    fig, ax = plt.subplots(figsize=(max(5, df.shape[1] * 0.9), max(3, df.shape[0] * 0.45)))
    im = ax.imshow(df.values, aspect="auto", cmap="RdBu_r",
                   norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=df.values.min(), vmax=df.values.max()))
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="SHAP value")
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _comparison_chart(shap_s: pd.Series, lime_s: pd.Series,
                      tshap_s: pd.Series | None, top_k: int = 8):
    """Side-by-side bar chart comparing SHAP, LIME, and T-SHAP (mean |value|)."""
    # Align to the union of features
    all_feat = list(dict.fromkeys(
        list(shap_s.index) + list(lime_s.index) +
        (list(tshap_s.index) if tshap_s is not None else [])
    ))
    shap_a  = shap_s.reindex(all_feat, fill_value=0.0).abs()
    lime_a  = lime_s.reindex(all_feat, fill_value=0.0).abs()

    # Union top-k by max across methods
    union   = (shap_a + lime_a)
    if tshap_s is not None:
        tshap_a = tshap_s.reindex(all_feat, fill_value=0.0).abs()
        union  += tshap_a
    else:
        tshap_a = None

    top_feat = union.nlargest(top_k).index.tolist()

    fig, ax  = plt.subplots(figsize=(9, max(3.5, len(top_feat) * 0.5)))
    y        = np.arange(len(top_feat))
    methods  = ["SHAP", "LIME"]
    colors   = ["#4C72B0", "#55A868"]
    data     = [shap_a.reindex(top_feat, fill_value=0.0).values,
                lime_a.reindex(top_feat, fill_value=0.0).values]

    if tshap_a is not None:
        methods.append("T-SHAP (mean)")
        colors.append("#C44E52")
        data.append(tshap_a.reindex(top_feat, fill_value=0.0).values)

    n   = len(methods)
    bw  = 0.7 / n
    for i, (method, color, vals) in enumerate(zip(methods, colors, data)):
        offset = (i - n / 2 + 0.5) * bw
        ax.barh(y + offset, vals, bw * 0.9, label=method, color=color, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(top_feat[::-1] if False else top_feat, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("|Importance|")
    ax.set_title("SHAP vs LIME vs T-SHAP — Feature Importance Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    ax.axvline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ===========================================================================
# Main render function
# ===========================================================================

def render():
    st.header("Explainability (SHAP / LIME / T-SHAP)")

    # ------------------------------------------------------------------
    # Sidebar-style controls
    # ------------------------------------------------------------------
    col_left, col_right = st.columns([1, 2])

    with col_left:
        model_type = st.selectbox("Select Model", _ALL_MODELS)

        uploaded_file = st.file_uploader("Upload Test Sample (CSV)", type=["csv"])
        if uploaded_file:
            raw_df = pd.read_csv(uploaded_file).iloc[:1]
        else:
            if os.path.exists(_DEFAULT_CSV):
                raw_df = pd.read_csv(_DEFAULT_CSV, nrows=1)
                st.info("Using first row of default dataset.")
            else:
                st.error(f"Default dataset not found: `{_DEFAULT_CSV}`. Upload a CSV.")
                return

        sample_clean = _clean_sample(raw_df)

        shap_n_bg    = st.slider("SHAP background samples", 10, 100, 30, step=10)
        lime_samples = st.slider("LIME neighbourhood samples", 100, 1000, 300, step=100)
        top_k        = st.slider("Features to display (top-k)", 3, 15, 8)

        run_btn = st.button("Predict & Explain", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Sample Preview")
        st.dataframe(sample_clean, use_container_width=True)

    if not run_btn:
        st.info("Configure options on the left, then click **Predict & Explain**.")
        return

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    with st.spinner(f"Loading {model_type} model..."):
        model = _load_model(model_type)
    if model is None:
        return

    # Align feature dimensions between sample and model
    if model_type == "LNN":
        n_needed = torch.load(_MODEL_PATHS["LNN"], weights_only=True)["input_layer.weight"].shape[1]
        sample_aligned = _align_features(sample_clean, n_needed)
    elif model_type == "RF":
        sample_aligned = sample_clean   # RF handles any width internally
    elif model_type in ("LSTM", "GRU", "RNN", "Transformer"):
        key_map = {"LSTM": "lstm.weight_ih_l0", "GRU": "gru.weight_ih_l0",
                   "RNN": "rnn.weight_ih_l0", "Transformer": "embedding.weight"}
        sd = torch.load(_MODEL_PATHS[model_type], weights_only=True)
        n_needed = sd[key_map[model_type]].shape[1]
        sample_aligned = _align_features(sample_clean, n_needed)
    else:
        sample_aligned = sample_clean

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    with st.spinner("Running inference..."):
        try:
            pred = _predict(model, sample_aligned, model_type)
        except Exception as exc:
            st.error(f"Inference error: {exc}")
            return

    if pred == 0:
        st.success("Prediction: Normal traffic (label = 0)")
    else:
        st.error("Prediction: Attack / Worm detected (label = 1)")

    st.divider()

    # ------------------------------------------------------------------
    # Generate all explanations
    # ------------------------------------------------------------------
    is_seq  = model_type in _SEQ_MODELS

    progress = st.progress(0, "Computing SHAP values...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_result = explain_with_shap(
            sample_aligned, model, model_type, n_bg=shap_n_bg
        )
    progress.progress(40, "Computing LIME values...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lime_result = explain_with_lime(
            sample_aligned, model, model_type,
            n_bg=shap_n_bg, window_size=_WINDOW_SIZE,
            num_features=top_k, num_samples=lime_samples,
        )
    progress.progress(75, "Computing T-SHAP (sequence)..." if is_seq else "T-SHAP skipped (flat model)...")

    tshap_df = None
    tshap_mean = None
    if is_seq:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tshap_df = explain_with_tshap(
                sample_aligned, model, model_type,
                window_size=_WINDOW_SIZE, n_bg=min(shap_n_bg, 20),
            )
        tshap_mean = tshap_df.abs().mean(axis=0)   # mean |shap| across timesteps

    progress.progress(100, "Done.")
    progress.empty()

    # ------------------------------------------------------------------
    # Display in tabs
    # ------------------------------------------------------------------
    if is_seq:
        tabs = st.tabs(["SHAP", "LIME", "T-SHAP (timestep)", "Comparison", "Liquid Parameters"])
    else:
        tabs = st.tabs(["SHAP", "LIME", "Comparison", "Liquid Parameters"])

    # ---- SHAP tab ----
    with tabs[0]:
        st.subheader(f"SHAP Feature Importance — {model_type}")
        if model_type == "RF":
            method_note = (
                "Method: TreeExplainer (exact Shapley values). "
                "Fallback: RF feature_importances_ (Gini) if model predicts constant output."
            )
        elif model_type in ("LNN",):
            method_note = "Method: KernelExplainer (model-agnostic, background data masking)"
        else:
            method_note = "Method: GradientExplainer (gradient x input, averaged over background)"
        st.caption(method_note)

        if shap_result.abs().sum() < 1e-10:
            st.warning(
                "All SHAP values are zero. This typically means the model predicts a "
                "constant output for this sample and background combination. "
                "Try re-training the model with balanced class data (use `python rf_baseline.py` "
                "or the full training scripts instead of the GUI quick-train)."
            )
        _bar_chart(shap_result, f"SHAP — Top features ({model_type})",
                   color="#4C72B0", top_k=top_k)

        with st.expander("Full SHAP values table"):
            st.dataframe(
                shap_result.rename("SHAP value").to_frame()
                    .style.background_gradient(cmap="RdBu_r", axis=0),
                use_container_width=True,
            )

    # ---- LIME tab ----
    with tabs[1]:
        st.subheader(f"LIME Feature Importance — {model_type}")
        st.caption("Method: LimeTabularExplainer (local linear surrogate, signed coefficients)")
        if lime_result.abs().sum() < 1e-10:
            st.warning(
                "All LIME values are zero. This typically means the model's predict_fn "
                "returns constant probabilities for all perturbed neighbourhood samples. "
                "Re-train the model with more diverse data to get meaningful explanations."
            )
        _bar_chart(lime_result, f"LIME — Top features ({model_type})",
                   color="#55A868", top_k=top_k)
        with st.expander("Full LIME values table"):
            st.dataframe(
                lime_result.rename("LIME coeff").to_frame()
                    .style.background_gradient(cmap="RdBu_r", axis=0),
                use_container_width=True,
            )

    # ---- T-SHAP tab (only for sequence models) ----
    if is_seq:
        with tabs[2]:
            st.subheader(f"T-SHAP — Timestep × Feature Heatmap ({model_type})")
            st.caption(
                "Method: GradientExplainer on full (1, window_size, features) input. "
                "Each cell shows how much that feature at that timestep pushed the prediction."
            )
            _heatmap_chart(tshap_df, f"T-SHAP heatmap — {model_type}")

            st.subheader("Mean |T-SHAP| per Feature (collapsed across time)")
            _bar_chart(tshap_mean, "Mean |T-SHAP| across timesteps",
                       color="#C44E52", top_k=top_k)

            with st.expander("T-SHAP values table (timestep x feature)"):
                st.dataframe(tshap_df.style.background_gradient(cmap="RdBu_r", axis=None),
                             use_container_width=True)

        comparison_tab_idx = 3
        liq_tab_idx        = 4
    else:
        comparison_tab_idx = 2
        liq_tab_idx        = 3

    # ---- Comparison tab ----
    with tabs[comparison_tab_idx]:
        st.subheader("SHAP vs LIME vs T-SHAP — Method Comparison")

        # Method guidance table
        guidance = pd.DataFrame({
            "Method":     ["SHAP (Tree)",      "SHAP (Kernel)",   "SHAP (Gradient)", "LIME",           "T-SHAP"],
            "Best for":   ["RF",               "LNN",             "LSTM/GRU/RNN/Transformer", "All models", "Sequence models"],
            "Speed":      ["Very fast",        "Slow",            "Fast",            "Medium",         "Fast"],
            "Fidelity":   ["Exact",            "Approx.",         "Approx.",         "Local approx.",  "Approx."],
            "Temporal":   ["No",               "No",              "No",              "No",             "Yes"],
        })
        st.dataframe(guidance, use_container_width=True, hide_index=True)

        if model_type == "RF":
            st.info("T-SHAP is not applicable for Random Forest (no temporal sequence).")
            tshap_mean = None
        elif not is_seq:
            st.info("T-SHAP is only available for sequence models (LSTM, GRU, RNN, Transformer, AR).")
            tshap_mean = None

        _comparison_chart(shap_result, lime_result, tshap_mean, top_k=top_k)

        # Narrative insight
        top_shap = shap_result.abs().idxmax() if shap_result.abs().sum() > 0 else "—"
        top_lime = lime_result.abs().idxmax() if lime_result.abs().sum() > 0 else "—"
        st.info(
            f"**Insight for {model_type}**\n\n"
            f"- Most important feature (SHAP): **{top_shap}**\n"
            f"- Most important feature (LIME): **{top_lime}**\n"
            + (f"- Most important feature (T-SHAP): **{tshap_mean.idxmax()}**\n" if tshap_mean is not None and tshap_mean.sum() > 0 else "")
            + (f"\n- **RNN** is the simplest recurrent baseline — no gating, no dropout. "
               f"LNN / AR / GAN are fastest for IoT edge deployment." if model_type in ("RNN", "LNN", "AR", "GAN") else "")
        )

    # ---- Liquid Parameters tab ----
    with tabs[liq_tab_idx]:
        st.header("LNN Liquid Parameters Analysis")

        if st.button("Generate Liquid Parameter Visualizations"):
            if not os.path.exists("models/lnn_model.pth"):
                st.error("LNN model not found. Please train it first.")
            else:
                with st.spinner("Analyzing liquid dynamics..."):
                    try:
                        from visualize_liquid_params import (
                            plot_liquid_gates_over_time,
                            plot_hidden_evolution_heatmap,
                            compare_lnn_vs_lstm,
                        )
                        from src.lstm_model import LSTMClassifier

                        sd_lnn    = torch.load("models/lnn_model.pth", weights_only=True)
                        input_dim = sd_lnn["input_layer.weight"].shape[1]
                        lnn_model = LNN(input_dim=input_dim, hidden_dim=16, num_steps=6, dt=0.1)
                        lnn_model.load_state_dict(sd_lnn)
                        lnn_model.eval()

                        lstm_model = None
                        if os.path.exists("models/lstm_model.pth"):
                            sd_lstm    = torch.load("models/lstm_model.pth", weights_only=True)
                            ldim       = sd_lstm["lstm.weight_ih_l0"].shape[1]
                            lstm_model = LSTMClassifier(input_dim=ldim, hidden_size=64)
                            lstm_model.load_state_dict(sd_lstm)
                            lstm_model.eval()

                        lnn_sample = _align_features(sample_clean, input_dim)
                        feat_vals  = lnn_sample.values.astype(np.float32)
                        X_seq      = torch.tensor(feat_vals, dtype=torch.float32)

                        plot_liquid_gates_over_time(lnn_model, X_seq)
                        plot_hidden_evolution_heatmap(lnn_model, X_seq)
                        compare_lnn_vs_lstm(lnn_model, lstm_model, X_seq)

                        st.success("Visualizations generated!")

                        t1, t2, t3 = st.tabs(["Gate Evolution", "Hidden State", "LNN vs LSTM"])
                        with t1:
                            if os.path.exists("figures/lnn_liquid_parameters.png"):
                                st.image("figures/lnn_liquid_parameters.png",
                                         caption="LNN Gate Evolution Over Time")
                            st.info("Decay and input gates dynamically adapt at each ODE step.")
                        with t2:
                            if os.path.exists("figures/lnn_hidden_heatmap.png"):
                                st.image("figures/lnn_hidden_heatmap.png",
                                         caption="LNN Hidden State Evolution Heatmap")
                            st.info("Each hidden unit evolves continuously during processing.")
                        with t3:
                            if os.path.exists("figures/lnn_vs_lstm_parameters.png"):
                                st.image("figures/lnn_vs_lstm_parameters.png",
                                         caption="LNN vs LSTM Parameter Comparison")
                            st.info("LNN liquid parameters change over time, enabling adaptive "
                                    "behaviour for zero-day worms (unlike LSTM's fixed weights).")
                    except Exception as exc:
                        st.error(f"Visualization failed: {exc}")
