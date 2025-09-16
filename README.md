# 🩺 Diabetes Prediction App (Random Forest)

This Streamlit app reproduces your notebook's Random Forest diabetes predictor using the same preprocessing and top-5 features.

## Features used
- HbA1c_level
- blood_glucose_level
- bmi
- age
- smoking_history (label-encoded)

## Project structure
- `app.py` — Streamlit app entrypoint
- `requirements.txt` — Python dependencies
- `README.md` — This file

## Run locally
1. Create and activate a virtual environment (optional but recommended).
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```
4. In the app, upload `diabetes_prediction_dataset.csv` (from Kaggle) or configure Kaggle credentials to auto-download (see below).

## Kaggle auto-download (optional)
If you set environment variables before running the app, it can fetch the dataset automatically:
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

On Windows PowerShell:
```powershell
$env:KAGGLE_USERNAME="your_kaggle_username"
$env:KAGGLE_KEY="your_kaggle_api_key"
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io/ (Streamlit Community Cloud) and sign in.
3. Create a new app:
   - Select the repo, branch, and set `app.py` as the entry file.
   - Add the following secrets if you want Kaggle auto-download:
     ```
     KAGGLE_USERNAME = your_kaggle_username
     KAGGLE_KEY = your_kaggle_api_key
     ```
4. Deploy. You'll get a public URL to share.

## Notes matching the notebook
- RandomForestClassifier with `n_estimators=200`, `random_state=42`.
- Train/Val/Test split and SMOTE used to explore class imbalance. Final model is fit on combined train+val as done in the notebook section.
- Decision threshold defaults to `0.89` for classification as in the notebook.
