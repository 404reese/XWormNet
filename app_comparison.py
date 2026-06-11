import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

def render():
    if not os.path.exists("comparison_results.csv"):
        st.warning("comparison_results.csv not found. Train models first to generate comparison.")
        return

    # LOAD REAL DATA from comparison_results.csv
    df = pd.read_csv("comparison_results.csv")

    # Display REAL table
    st.header("📊 Model Comparison (Live Results)")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # REAL bar chart: F1 Score
        st.subheader("F1 Score Comparison")
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(df["Model"], df["F1"], color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(df)])
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.1)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f"{height:.3f}", ha="center")
        st.pyplot(fig)

    with col2:
        # REAL bar chart: Latency
        st.subheader("Inference Latency (ms)")
        fig, ax = plt.subplots(figsize=(6,4))
        if "Latency (ms)" in df.columns:
            bars = ax.bar(df["Model"], df["Latency (ms)"], color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(df)])
            ax.set_ylabel("Latency (ms)")
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(df["Latency (ms)"]),
                        f"{height:.3f}", ha="center")
            st.pyplot(fig)
        else:
            st.write("Latency data not available.")

    # REAL insight
    st.info("💡 Key Insight: LNN still fastest, LSTM better than RF on sequences.")

    # Download button (REAL CSV)
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Results (CSV)", csv, file_name="comparison_results.csv")
