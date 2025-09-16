import os
import io
import zipfile
import tempfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# -----------------------------------------
# App Config
# -----------------------------------------
st.set_page_config(page_title="Diabetes Prediction (Random Forest)", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("""
This app reproduces your notebook's Random Forest diabetes predictor.

Training expects the public Kaggle dataset "diabetes_prediction_dataset" or a CSV with the same columns.

Selected features used by the model:
- HbA1c_level
- blood_glucose_level
- bmi
- age
- smoking_history (label-encoded)
""")

# -----------------------------------------
# Data loading helpers
# -----------------------------------------
DATASET_SLUG = "suraj520/diabetes-prediction-dataset"  # mirrors the Kaggle path your notebook used

REQUIRED_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes",
]

SELECTED_FEATURES = [
    "HbA1c_level",
    "blood_glucose_level",
    "bmi",
    "age",
    "smoking_history",
]

@st.cache_data(show_spinner=False)
def load_csv_from_upload(upload) -> pd.DataFrame:
    return pd.read_csv(upload)

@st.cache_data(show_spinner=True)
def load_from_kaggle() -> Optional[pd.DataFrame]:
    """Attempt to download via Kaggle API if credentials are available.
    Requires environment variables: KAGGLE_USERNAME, KAGGLE_KEY
    """
    try:
        import kaggle  # type: ignore
    except Exception:
        return None

    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download the dataset zip to tmpdir
            # Note: this command mirrors 'kaggle datasets download -d <slug>'
            api = kaggle.api
            api.authenticate()
            api.dataset_download_files(DATASET_SLUG, path=tmpdir, quiet=True)

            # Find and read the CSV from the downloaded zip
            zips = [f for f in os.listdir(tmpdir) if f.endswith('.zip')]
            if not zips:
                return None
            zip_path = os.path.join(tmpdir, zips[0])

            with zipfile.ZipFile(zip_path, 'r') as zf:
                csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
                if not csv_names:
                    return None
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
            return df
    except Exception:
        return None

# -----------------------------------------
# Training helper
# -----------------------------------------
@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, LabelEncoder, list]:
    """Train a RandomForest model reproducing your notebook's key steps.

    Returns (model, label_encoder_for_smoking, feature_order)
    """
    # Basic validation
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Label encode categorical columns as in the notebook
    df_encoded = df.copy()
    label_cols = ['gender', 'smoking_history']
    le_smoke = LabelEncoder()
    le_gender = LabelEncoder()

    df_encoded['smoking_history'] = le_smoke.fit_transform(df_encoded['smoking_history'].astype(str))
    df_encoded['gender'] = le_gender.fit_transform(df_encoded['gender'].astype(str))

    # Use the same selected top-5 features determined in the notebook
    X = df_encoded.drop('diabetes', axis=1)
    y = df_encoded['diabetes']

    X_selected = X[SELECTED_FEATURES].copy()

    # Split (train/val/test) like the notebook, then SMOTE on the training part only
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # SMOTE ratio to address imbalance similar to the notebook approach
    n_maj = (y_train == 0).sum()
    n_min = (y_train == 1).sum()
    sampling_ratio = max(n_min, 1) / max(n_maj, 1)
    sm = SMOTE(sampling_strategy=sampling_ratio, random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Final model config per notebook (rf_final): n_estimators=200, random_state=42
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
    )

    # Train on combined train+val like the notebook did (after initial selection)
    # To keep label encoding consistent and simple, refit on concatenated original X (without SMOTE) as done in notebook
    X_train_final = np.concatenate([X_train.values, X_val.values], axis=0)
    y_train_final = np.concatenate([y_train.values, y_val.values], axis=0)
    model.fit(X_train_final, y_train_final)

    feature_order = SELECTED_FEATURES
    return model, le_smoke, feature_order

# -----------------------------------------
# Load data (Kaggle or upload)
# -----------------------------------------
with st.expander("1) Provide training data (optional if already deployed with data)"):
    st.write("Provide the dataset to train the model. Options:")
    st.markdown("- Upload the CSV manually.")
    st.markdown("- Or, if this app is deployed with Kaggle credentials, it can auto-download.")

    upload = st.file_uploader("Upload diabetes_prediction_dataset.csv", type=["csv"]) 
    df: Optional[pd.DataFrame] = None

    if upload is not None:
        try:
            df = load_csv_from_upload(upload)
            st.success(f"Loaded CSV with shape {df.shape}")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if df is None:
        df = load_from_kaggle()
        if df is not None:
            st.success(f"Downloaded from Kaggle: shape {df.shape}")
        else:
            st.info("Kaggle download not available. Please upload the CSV.")

# -----------------------------------------
# Train or recall cached model
# -----------------------------------------
model: Optional[RandomForestClassifier] = None
le_smoke: Optional[LabelEncoder] = None
feature_order: Optional[list] = None

if df is not None:
    try:
        with st.spinner("Training model (first run is cached)..."):
            model, le_smoke, feature_order = train_model(df)
        st.success("Model trained and cached.")
    except Exception as e:
        st.error(f"Training failed: {e}")

# -----------------------------------------
# Prediction UI
# -----------------------------------------
st.header("2) Predict for a patient")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.3, step=0.1, format="%.1f")
    hba1c = st.number_input("HbA1c level", min_value=3.0, max_value=20.0, value=5.6, step=0.1, format="%.1f")
with col2:
    glucose = st.number_input("Blood glucose level", min_value=50, max_value=400, value=140, step=1)
    smoking_cat = st.selectbox(
        "Smoking history",
        options=[
            "never",
            "No Info",
            "former",
            "current",
            "ever",
            "not current",
        ],
        index=0,
    )

st.caption("Note: Only the 5 selected features are used by the model as in your notebook.")

predict_btn = st.button("Predict Diabetes Risk")

if predict_btn:
    if model is None or le_smoke is None or feature_order is None:
        st.error("Model not available yet. Please provide the dataset to train first.")
    else:
        # Transform smoking history with the same encoder that was fit during training
        # If the chosen category wasn't seen during fit, fall back gracefully
        try:
            smoking_val = le_smoke.transform([smoking_cat])[0]
        except Exception:
            # unseen label: map to nearest known by string
            known = list(le_smoke.classes_)
            smoking_val = le_smoke.transform([known[0]])[0]

        # Build a single-row DataFrame matching the training feature order
        x_row = pd.DataFrame([[hba1c, glucose, bmi, age, smoking_val]], columns=feature_order)

        # Probability of class 1 (diabetes)
        prob = float(model.predict_proba(x_row)[0, 1])

        # Threshold used in notebook examples; adjust if needed
        threshold = 0.89
        pred = int(prob >= threshold)

        st.subheader("Result")
        st.metric("Predicted probability", f"{prob:.3f}")
        st.write("Decision threshold:", threshold)
        if pred == 1:
            st.error("Model prediction: Diabetes (positive)")
        else:
            st.success("Model prediction: No Diabetes (negative)")

st.divider()
st.markdown("""
### Notes
- This app mirrors the RandomForest configuration from your notebook (`n_estimators=200`, `random_state=42`).
- It uses the 5 most important features discovered in your notebook for training and prediction.
- To deploy on Streamlit Community Cloud, push these files to a public GitHub repo and create a new Streamlit app pointing to `app.py`.
""")

