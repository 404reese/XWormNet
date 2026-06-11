import streamlit as st

def render():
    st.header("🏠 Home")
    st.markdown("""
    Welcome to the **XWormNet Dashboard**, an Explainable Liquid Neural Network (LNN) pipeline for Zero-Day Worm Detection.
    
    ### Features:
    - **📊 Model Comparison**: View and compare live results from LNN and Random Forest models.
    - **🎯 Train Model**: Train new models on the fly using your dataset.
    - **🔍 Explainability**: Generate SHAP and LIME explanations for predictions.
    - **💾 Export**: Download comparison results in CSV or markdown formats.
    
    Use the sidebar on the left to navigate through the dashboard.
    """)
