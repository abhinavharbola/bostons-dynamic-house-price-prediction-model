import streamlit as st
import pandas as pd
import joblib
import json

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏡",
    layout="centered"
)

# ----------------------------
# Load Model and Artifacts
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
def load_artifacts():
    """Loads the artifacts (column info, stats) for the UI."""
    try:
        with open('model_artifacts.json') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Artifacts file (model_artifacts.json) not found. Please run train.py first.")
        return None

model = load_model()
artifacts = load_artifacts()

# ----------------------------
# Header
# ----------------------------
st.markdown("""
    <h1 style='text-align:center; color:#6C63FF;'>🏡 Boston House Price Dynamics</h1>
    <p style='text-align:center; color:gray;'>Predict the median value of homes using a Random Forest Regressor</p>
    <hr style='border:1px solid #eee;'>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar (User Input)
# ----------------------------
st.sidebar.header("⚙️ Specify Input Parameters")

# Tooltips for features (manually maintained)
feature_info = {
    "CRIM": "Per capita crime rate by town",
    "ZN": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
    "INDUS": "Proportion of non-retail business acres per town",
    "CHAS": "Charles River dummy variable (1 if tract bounds river; 0 otherwise)",
    "NOX": "Nitric oxides concentration (parts per 10 million)",
    "RM": "Average number of rooms per dwelling",
    "AGE": "Proportion of owner-occupied units built prior to 1940",
    "DIS": "Weighted distances to five Boston employment centers",
    "RAD": "Index of accessibility to radial highways",
    "TAX": "Full-value property-tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "LSTAT": "% lower status of the population"
}

def user_input_features():
    """
    Creates sidebar controls dynamically from loaded artifacts.
    This prevents data leakage by using only training set statistics.
    """
    if not artifacts:
        return None

    features = {}

    # Create controls for categorical features
    for col, props in artifacts['categorical_features'].items():
        features[col] = st.sidebar.selectbox(
            label=col,
            options=props['options'],
            help=feature_info.get(col, "")
        )

    # Create controls for numeric features
    for col, props in artifacts['numeric_features'].items():
        features[col] = st.sidebar.slider(
            label=col,
            min_value=props['min'],
            max_value=props['max'],
            value=props['mean'],
            help=feature_info.get(col, "")
        )

    # Re-order features to match model's training order
    ordered_features = {col: features[col] for col in artifacts['column_order']}
    return pd.DataFrame(ordered_features, index=[0])

# Only proceed if model and artifacts are loaded
if model and artifacts:
    df = user_input_features()

    # ----------------------------
    # Main Page (Prediction)
    # ----------------------------
    st.subheader("Specified Input Parameters")
    
    # --- THIS IS THE FIX ---
    # Replaced 'use_container_width=True' with 'width="stretch"'
    st.dataframe(df, width="stretch")
    # -----------------------

    prediction = model.predict(df)[0]

    st.markdown(f"""
        <div style="
            background-color:#E9F5FF;
            padding:25px;
            border-radius:20px;
            text-align:center;
            box-shadow:0 4px 15px rgba(0,0,0,0.1);
            font-size:24px;
            font-weight:bold;
            color:#333;
            margin-top:20px;">
            💵 <span style='color:#111;'>Predicted Median House Price:</span>
            <br><span style='font-size:30px; color:#0B5ED7;'>${prediction*1000:,.2f}</span>
        </div>
    """, unsafe_allow_html=True)

    st.info("Navigate to the **Shap Graph** page to see feature importance.")

    with st.expander("Show Model Performance (from test set)"):
        st.json(artifacts['evaluation_metrics'])
else:
    st.warning("Model and/or artifacts not loaded. Please run `python train.py` to generate them.")