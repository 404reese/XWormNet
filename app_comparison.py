import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

# 8-model colour palette (one per model, consistent across charts)
MODEL_COLORS = {
    "LNN":         "#4C72B0",
    "RF":          "#DD8452",
    "LSTM":        "#55A868",
    "GRU":         "#C44E52",
    "Transformer": "#8172B2",
    "GAN":         "#937860",
    "AR":          "#DA8BC3",
    "RNN":         "#8C8C8C",
}

def render():
    if not os.path.exists("outputs/csv/comparison_results.csv"):
        st.warning(
            "comparison_results.csv not found. "
            "Run `python compare_models.py` (or train all 8 models via the GUI) first."
        )
        return

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    df = pd.read_csv("outputs/csv/comparison_results.csv")

    st.header(":material/bar_chart: Model Comparison — 8-Model Results")

    # ----------------------------------------------------------------
    # Full results table
    # ----------------------------------------------------------------
    st.subheader("📋 Results Table")
    st.dataframe(df.style.format({
        "Precision":    "{:.4f}",
        "Recall":       "{:.4f}",
        "F1":           "{:.4f}",
        "Latency (ms)": "{:.4f}",
        "Size (MB)":    "{:.4f}",
    }), use_container_width=True)

    # ----------------------------------------------------------------
    # Charts (two-column layout)
    # ----------------------------------------------------------------
    col1, col2 = st.columns(2)

    colors = [MODEL_COLORS.get(m, "#999999") for m in df["Model"]]

    with col1:
        st.subheader("F1 Score Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(df["Model"], df["F1"], color=colors)
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.12)
        ax.tick_params(axis="x", rotation=30)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01,
                    f"{h:.3f}", ha="center", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Inference Latency (ms / sample)")
        if "Latency (ms)" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(df["Model"], df["Latency (ms)"], color=colors)
            ax.set_ylabel("Latency (ms)")
            ax.tick_params(axis="x", rotation=30)
            max_lat = df["Latency (ms)"].max()
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        h + 0.01 * max_lat,
                        f"{h:.3f}", ha="center", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.write("Latency data not available.")

    # Second row: Precision vs Recall + Size
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Precision vs Recall")
        if "Precision" in df.columns and "Recall" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            for _, row in df.iterrows():
                c = MODEL_COLORS.get(row["Model"], "#999999")
                ax.scatter(row["Recall"], row["Precision"], color=c, s=100, zorder=3)
                ax.annotate(row["Model"], (row["Recall"], row["Precision"]),
                            textcoords="offset points", xytext=(5, 3), fontsize=8)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_xlim(0, 1.1)
            ax.set_ylim(0, 1.1)
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig)

    with col4:
        st.subheader("Model Size (MB)")
        if "Size (MB)" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(df["Model"], df["Size (MB)"], color=colors)
            ax.set_ylabel("Size (MB)")
            ax.tick_params(axis="x", rotation=30)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.001,
                        f"{h:.3f}", ha="center", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)

    # ----------------------------------------------------------------
    # Insights
    # ----------------------------------------------------------------
    fastest = df.loc[df["Latency (ms)"].idxmin(), "Model"]
    top_f1  = df.loc[df["F1"].idxmax(), "Model"]
    smallest = df.loc[df["Size (MB)"].idxmin(), "Model"]

    st.info(
        f"**Key Insights (8-model comparison)**\n\n"
        f"- 🏆 **Highest F1**: {top_f1}\n"
        f"- ⚡ **Fastest inference**: {fastest} — RNN is simplest recurrent baseline; "
        f"LNN / AR / GAN are the fastest for edge IoT deployment\n"
        f"- 📦 **Smallest model**: {smallest}\n"
        f"- RNN (vanilla Elman RNN, no gating) serves as the minimal recurrent lower-bound "
        f"baseline vs. LSTM & GRU in the IEEE paper comparison.",
        icon=":material/lightbulb:"
    )

    # ----------------------------------------------------------------
    # Download button
    # ----------------------------------------------------------------
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Results (CSV)",
        csv,
        file_name="comparison_results.csv",
        icon=":material/download:"
    )
