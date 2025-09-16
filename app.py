import os
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

# ---------- Sidebar Settings ----------
st.sidebar.header("Settings")
st.sidebar.caption("Adjust global app settings")
# Language toggle
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"], index=0)
is_ar = (lang == "العربية")
# Clinician mode toggle
clinician_mode = st.sidebar.toggle("Clinician mode", value=False, help="Show evaluation metrics and curves (for professionals)")
# Threshold slider
threshold_sidebar = st.sidebar.slider(
    "Decision threshold" if not is_ar else "عتبة القرار",
    min_value=0.50, max_value=0.99, value=0.89, step=0.01
)
st.sidebar.markdown(
    ("The decision threshold converts probability to positive/negative. Higher threshold = fewer positives.")
    if not is_ar else ("عتبة القرار تحول الاحتمالية إلى قرار نهائي. كلما زادت العتبة قلّت الإيجابيات المتوقعة.")
)

# ---------- App Title & Intro (Hero) ----------
st.markdown(
    """
    <div style="padding: 18px; border-radius: 14px; background: linear-gradient(135deg,#f0f7ff 0%, #ffffff 100%); border:1px solid #e6e9ef;">
      <div style="display:flex; align-items:center; gap:14px;">
        <div style="font-size:36px;">🩺</div>
        <div>
          <div style="font-size:24px; font-weight:700;">Diabetes Risk Checker</div>
          <div style="color:#4c566a;">A patient-friendly tool powered by a Random Forest model.</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Simple CSS for card-like containers and badges
st.markdown(
    """
    <style>
    /* Theme-aware cards and badges */
    .card { border-radius:12px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,0.04); }
    /* Light mode */
    @media (prefers-color-scheme: light) {
      .card { background:#ffffff; border:1px solid #e6e9ef; color:#111827; }
      .badge-low { background:#e8f5e9; color:#2e7d32; border:1px solid #c8e6c9; }
      .badge-med { background:#fff8e1; color:#8d6e00; border:1px solid #ffe082; }
      .badge-high{ background:#ffebee; color:#b71c1c; border:1px solid #ffcdd2; }
    }
    /* Dark mode */
    @media (prefers-color-scheme: dark) {
      .card { background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.15); color:#ECEFF4; }
      .badge-low { background:rgba(46,125,50,0.15); color:#A3E635; border:1px solid rgba(163,230,53,0.35); }
      .badge-med { background:rgba(255,193,7,0.15); color:#FFD166; border:1px solid rgba(255,209,102,0.35); }
      .badge-high{ background:rgba(183,28,28,0.18); color:#FF8B8B; border:1px solid rgba(255,139,139,0.35); }
    }
    .card-title { font-weight:600; margin-bottom:8px; }
    .badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; font-size:0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def risk_badge(prob: float) -> str:
    if prob < 0.33:
        return '<span class="badge badge-low">Low risk</span>'
    elif prob < 0.66:
        return '<span class="badge badge-med">Moderate risk</span>'
    else:
        return '<span class="badge badge-high">High risk</span>'

# -----------------------------------------
# Data & features config
# -----------------------------------------
DEFAULT_CSV_PATH = r"C:\\Users\\Youssef\\OneDrive\\Desktop\\semi project\\diabetes_prediction_dataset.csv"

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

@st.cache_data(show_spinner=False)
def load_local_csv(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
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

    # SMOTE ratio to address imbalance (ensure we only oversample, not downsample the minority)
    n_maj = (y_train == 0).sum()
    n_min = (y_train == 1).sum()
    if n_min == 0 or n_maj == 0:
        # لا يمكن تطبيق SMOTE إذا كانت إحدى الفئات صفر
        X_train_res, y_train_res = X_train, y_train
    else:
        # توازن كامل 1.0 يعني بعد الـ SMOTE: minority == majority
        try:
            sm = SMOTE(sampling_strategy=1.0, random_state=42)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        except Exception:
            # إذا فشل SMOTE لأي سبب، نكمل بدون إعادة توليد عينات
            X_train_res, y_train_res = X_train, y_train

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
# Load data (local path first, then manual upload)
# -----------------------------------------
with st.expander("Data & Model (maintainers)" if not is_ar else "البيانات والنموذج (للمطورين)", expanded=False):
    st.subheader("1) Data Source" if not is_ar else "١) مصدر البيانات")
    st.caption("The app will try to read the file automatically from this path:" if not is_ar else "سيتم محاولة قراءة الملف تلقائياً من هذا المسار:")
    st.code(DEFAULT_CSV_PATH, language="text")

    df: Optional[pd.DataFrame] = load_local_csv(DEFAULT_CSV_PATH)

    if df is not None:
        st.success((f"Found local dataset. Shape: {df.shape}") if not is_ar else (f"تم العثور على الملف محلياً. الشكل: {df.shape}"))
    else:
        st.info("Dataset not found at the specified path. You can upload a CSV with the expected columns." if not is_ar else "لم يتم العثور على الملف. يمكنك رفع ملف CSV بالأعمدة المطلوبة.")
        upload = st.file_uploader("Upload diabetes_prediction_dataset.csv" if not is_ar else "ارفع ملف diabetes_prediction_dataset.csv", type=["csv"]) 
        if upload is not None:
            try:
                df = load_csv_from_upload(upload)
                st.success((f"CSV uploaded successfully. Shape: {df.shape}") if not is_ar else (f"تم تحميل الملف بنجاح. الشكل: {df.shape}"))
            except Exception as e:
                st.error((f"Failed to read CSV: {e}") if not is_ar else (f"تعذر قراءة الملف: {e}"))

# -----------------------------------------
# Train or recall cached model
# -----------------------------------------
model: Optional[RandomForestClassifier] = None
le_smoke: Optional[LabelEncoder] = None
feature_order: Optional[list] = None
# For clinician mode evaluation
X_test_glob: Optional[np.ndarray] = None
y_test_glob: Optional[np.ndarray] = None

if df is not None:
    try:
        with st.spinner("Training model (first run is cached)..."):
            model, le_smoke, feature_order = train_model(df)
        st.success("Model trained and cached." if not is_ar else "تم تدريب النموذج وتخزينه.")
    except Exception as e:
        st.error((f"Training failed: {e}") if not is_ar else (f"فشل التدريب: {e}"))

tabs = st.tabs((["Overview", "Check Risk", "Education", "About"]) if not is_ar else (["نظرة عامة", "تحقق من الخطر", "تثقيف", "حول"]))

with tabs[0]:
    st.markdown(("""
    ### What is this?
    This tool estimates your diabetes risk using five factors your clinician may consider.
    It does not replace medical advice. Always consult a healthcare professional.
    """) if not is_ar else ("""
    ### ما هذا؟
    هذه الأداة تقدّر خطر السكري باستخدام خمسة عوامل. لا تغني عن الاستشارة الطبية.
    """))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(((
        """
        <div class="card">
          <div class="card-title">How it works</div>
          • Enter your information in the Check Risk tab.
          <br/>• The model outputs a probability and a suggested decision.
          <br/>• You can adjust the decision threshold from the sidebar.
        </div>
        """
        )) if not is_ar else (
        """
        <div class="card">
          <div class="card-title">طريقة العمل</div>
          • أدخل بياناتك في تبويب "تحقق من الخطر".
          <br/>• سيعرض النموذج احتمالية وقراراً مقترحاً.
          <br/>• يمكنك ضبط عتبة القرار من الشريط الجانبي.
        </div>
        """
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(((
        """
        <div class="card">
          <div class="card-title">Features used</div>
          HbA1c level, Blood glucose level, BMI, Age, Smoking history
        </div>
        """
        )) if not is_ar else (
        """
        <div class="card">
          <div class="card-title">الخصائص المستخدمة</div>
          HbA1c، سكر الدم، مؤشر كتلة الجسم، العمر، تاريخ التدخين
        </div>
        """
        ), unsafe_allow_html=True)

with tabs[1]:
    st.markdown(("#### Enter your details") if not is_ar else ("#### أدخل بياناتك"))
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🎂 Age" if not is_ar else "🎂 العمر", min_value=0, max_value=120, value=45, step=1)
        bmi = st.number_input("⚖️ BMI" if not is_ar else "⚖️ مؤشر كتلة الجسم", min_value=10.0, max_value=60.0, value=27.3, step=0.1, format="%.1f")
        hba1c = st.number_input("🧪 HbA1c level" if not is_ar else "🧪 مستوى HbA1c", min_value=3.0, max_value=20.0, value=5.6, step=0.1, format="%.1f")
    with col2:
        glucose = st.number_input("🩸 Blood glucose level" if not is_ar else "🩸 سكر الدم", min_value=50, max_value=400, value=140, step=1)
        smoking_cat = st.selectbox(
            ("🚬 Smoking history") if not is_ar else ("🚬 تاريخ التدخين"),
            options=["never", "No Info", "former", "current", "ever", "not current"],
            index=0,
        )

    st.caption(("Note: Only the 5 selected features are used by the model as in your notebook.") if not is_ar else ("ملاحظة: النموذج يستخدم 5 خصائص فقط كما في النوتبوك."))

    predict_btn = st.button("Predict Diabetes Risk" if not is_ar else "احسب خطر السكري", type="primary")

    if predict_btn:
        if model is None or le_smoke is None or feature_order is None:
            st.error(("Model not available yet. Please provide the dataset to train first.") if not is_ar else ("الموديل غير متاح بعد. وفّر بيانات التدريب أولاً."))
        else:
            try:
                smoking_val = le_smoke.transform([smoking_cat])[0]
            except Exception:
                known = list(le_smoke.classes_)
                smoking_val = le_smoke.transform([known[0]])[0]

            x_row = pd.DataFrame([[hba1c, glucose, bmi, age, smoking_val]], columns=feature_order)
            prob = float(model.predict_proba(x_row)[0, 1])
            threshold = float(threshold_sidebar)
            pred = int(prob >= threshold)

            st.markdown(("""
            <div class=\"card\">
              <div class=\"card-title\">Your result</div>
            """) if not is_ar else ("""
            <div class=\"card\">
              <div class=\"card-title\">نتيجتك</div>
            """), unsafe_allow_html=True)
            colA, colB = st.columns([1,1])
            with colA:
                st.metric(("Predicted probability") if not is_ar else ("الاحتمالية المتوقعة"), f"{prob:.3f}")
                st.write(("Decision threshold:") if not is_ar else ("عتبة القرار:"), threshold)
                st.progress(int(round(prob * 100)))
            with colB:
                st.markdown(risk_badge(prob), unsafe_allow_html=True)
                if pred == 1:
                    st.error(("Model prediction: Diabetes (positive)") if not is_ar else ("تنبؤ النموذج: سكري (إيجابي)"))
                else:
                    st.success(("Model prediction: No Diabetes (negative)") if not is_ar else ("تنبؤ النموذج: لا سكري (سلبي)"))
            st.markdown("</div>", unsafe_allow_html=True)

            # Optional downloadable mini report
            report = (
                (f"Age: {age}\nBMI: {bmi}\nHbA1c: {hba1c}\nBlood glucose: {glucose}\n" 
                 f"Smoking history: {smoking_cat}\n\nPredicted probability: {prob:.3f}\n" 
                 f"Decision threshold: {threshold}\nPrediction: {'Diabetes (positive)' if pred==1 else 'No Diabetes (negative)'}\n")
                if not is_ar else
                (f"العمر: {age}\nمؤشر كتلة الجسم: {bmi}\nHbA1c: {hba1c}\nسكر الدم: {glucose}\n" 
                 f"تاريخ التدخين: {smoking_cat}\n\nالاحتمالية: {prob:.3f}\n" 
                 f"عتبة القرار: {threshold}\nالنتيجة: {'سكري (إيجابي)' if pred==1 else 'لا سكري (سلبي)'}\n")
            )
            st.download_button(("Download mini report") if not is_ar else ("تحميل تقرير مختصر"), data=report, file_name="diabetes_risk_report.txt")

            # Session history
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].append({
                "age": age, "bmi": bmi, "hba1c": hba1c, "glucose": glucose,
                "smoking": smoking_cat, "prob": prob, "pred": pred, "threshold": threshold
            })
            with st.expander(("Recent assessments") if not is_ar else ("آخر التقييمات"), expanded=False):
                if st.session_state["history"]:
                    for i, item in enumerate(reversed(st.session_state["history"][-5:]), start=1):
                        st.write(f"{i}. Prob={item['prob']:.3f} | Pred={'Pos' if item['pred']==1 else 'Neg'} | Age={item['age']} | BMI={item['bmi']}")
                else:
                    st.write(("No assessments yet.") if not is_ar else ("لا توجد تقييمات بعد."))

with tabs[2]:
    st.markdown((("""
    #### Learn about diabetes risk
    • Maintain a healthy BMI and active lifestyle.
    
    • Monitor your blood glucose and HbA1c regularly if advised by your clinician.
    
    • Quit smoking; it improves overall cardiovascular and metabolic health.
    
    • This tool is for educational purposes and does not replace professional medical advice.
    """)) if not is_ar else ("""
    #### تعلّم عن مخاطر السكري
    • حافظ على مؤشر كتلة صحّي ونشاط بدني.
    
    • راقب سكر الدم وHbA1c حسب توجيه الطبيب.
    
    • الإقلاع عن التدخين يحسّن الصحة القلبية والاستقلابية.
    
    • هذه الأداة تعليمية ولا تغني عن الاستشارة الطبية.
    """))

with tabs[3]:
    st.markdown((("""
    #### About this app
    Built with Streamlit and a Random Forest model mirroring your notebook settings (n_estimators=200, random_state=42).
    The app uses the following features: HbA1c level, Blood glucose level, BMI, Age, and Smoking history.
    """)) if not is_ar else ("""
    #### نبذة عن التطبيق
    بُني باستخدام Streamlit ونموذج Random Forest مماثل لإعدادات النوتبوك (n_estimators=200, random_state=42).
    يستخدم التطبيق: HbA1c، سكر الدم، مؤشر كتلة الجسم، العمر، وتاريخ التدخين.
    """))

# ---------- Clinician Mode ----------
if clinician_mode and (model is not None) and (df is not None):
    st.divider()
    st.subheader("Clinician dashboard" if not is_ar else "لوحة المختص")
    # Recreate evaluation split for metrics (use the same split logic quickly)
    try:
        # Prepare encoded data similarly
        df_encoded = df.copy()
        df_encoded['smoking_history'] = LabelEncoder().fit_transform(df_encoded['smoking_history'].astype(str))
        df_encoded['gender'] = LabelEncoder().fit_transform(df_encoded['gender'].astype(str))
        X_full = df_encoded.drop('diabetes', axis=1)
        y_full = df_encoded['diabetes']
        X_sel = X_full[SELECTED_FEATURES].copy()
        X_train, X_temp, y_train, y_temp = train_test_split(X_sel, y_full, test_size=0.3, random_state=42, stratify=y_full)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        # Evaluate on X_test
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
        proba = model.predict_proba(X_test)[:,1]
        y_pred = (proba >= float(threshold_sidebar)).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_test, y_pred)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1-score", f"{f1:.3f}")

        # Plot ROC
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        ax.plot([0,1],[0,1], 'k--', alpha=0.4)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig, use_container_width=True)

        # Show confusion matrix
        st.write("Confusion matrix:")
        st.write(cm)
    except Exception as e:
        st.info(f"Clinician metrics unavailable: {e}")

st.divider()
st.markdown("""
### Notes
- The model uses `RandomForestClassifier` with (`n_estimators=200`, `random_state=42`).
- It relies on the same top 5 features from the notebook for both training and prediction.
- To deploy on Streamlit Community Cloud, push the project to a public GitHub repo and point to `app.py`.
""")

