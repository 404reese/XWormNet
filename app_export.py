import streamlit as st
import pandas as pd
import os

def render():
    st.header("💾 Export Results")

    comp_path = "outputs/csv/comparison_results.csv"
    if not os.path.exists(comp_path):
        st.warning(f"{comp_path} not found. Please train models first to generate comparison results.")
        return

    # LOAD COMPARISON (REAL)
    df = pd.read_csv(comp_path)

    # DOWNLOAD CSV (REAL)
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Comparison (CSV)", csv, file_name="comparison_results.csv")

    # PAPER-READY TABLE (REAL)
    st.subheader("📄 Paper-Ready Table (Markdown for IEEE)")
    markdown = df.to_markdown(index=False)
    st.code(markdown, language="markdown")

    # COPY BUTTON
    st.text_area("Copy Table", markdown, height=200)
