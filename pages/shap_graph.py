import streamlit as st
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd

st.set_page_config(page_title="Shap Graph: Boston House Price App", page_icon="📊")

# ----------------------------
# Load Model and Data
# ----------------------------
@st.cache_resource
def load_model():
    """Loads the pre-trained model."""
    try:
        return joblib.load('model.joblib')
    except FileNotFoundError:
        st.error("Model file (model.joblib) not found. Please run train.py first.")
        return None

@st.cache_data
def load_background_data():
    """Loads the training data used as the SHAP background."""
    try:
        # Load the saved X_train data
        return pd.read_parquet('X_train.parquet')
    except FileNotFoundError:
        st.error("Background data (X_train.parquet) not found. Please run train.py first.")
        return None

model = load_model()
X_train_background = load_background_data()

# ----------------------------
# Header
# ----------------------------
st.markdown("""
    <h1 style='text-align:center; color:#6C63FF;'>Explainability Dashboard</h1>
    <p style='text-align:center; color:gray;'>Understand which features drive Boston house prices</p>
    <hr style='border:1px solid #eee;'>
""", unsafe_allow_html=True)

# ----------------------------
# SHAP Analysis
# ----------------------------
if model and X_train_background is not None:
    st.subheader("🔹 SHAP Summary Plot")
    st.markdown("""
        This plot shows the most important features and their impact on the prediction.
        - **Feature Importance:** Features are ranked by importance (top to bottom).
        - **Impact:** The horizontal location shows whether that feature's value had a high (red) or low (blue) impact on the prediction.
    """)
    
    # Use st.cache_data for the expensive SHAP calculation
    @st.cache_data
    def calculate_shap_values(_model, _background_data):
        explainer = shap.TreeExplainer(_model)
        return explainer.shap_values(_background_data)

    shap_values = calculate_shap_values(model, X_train_background)

    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, X_train_background, show=False)
    st.pyplot(fig1, bbox_inches="tight")

    st.write("---")
    st.subheader("🔹 SHAP Bar Plot")
    st.markdown("This plot shows the average impact of each feature on the model's output magnitude (mean absolute SHAP value).")

    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, X_train_background, plot_type="bar", show=False)
    st.pyplot(fig2, bbox_inches="tight")

    st.success("You can go back to the Home page for predictions.")
else:
    st.warning("Model and/or background data not loaded. Please run `python train.py` to generate them.")
