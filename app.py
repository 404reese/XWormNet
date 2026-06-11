import streamlit as st
import app_home
import app_comparison
import app_train
import app_explainability
import app_export

st.set_page_config(layout="wide", page_title="XWormNet Dashboard")
st.title("🦠 XWormNet - Explainable LNN for Zero-Day Worm Detection")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Model Comparison", "🎯 Train Model", "🔍 Explainability", "💾 Export"])

if page == "🏠 Home":
    app_home.render()
elif page == "📊 Model Comparison":
    app_comparison.render()
elif page == "🎯 Train Model":
    app_train.render()
elif page == "🔍 Explainability":
    app_explainability.render()
elif page == "💾 Export":
    app_export.render()
