# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# --------------------------
# Page config + style
# --------------------------
st.set_page_config(page_title="Diabetes Prediction Dashboard",
                   page_icon="🩺",
                   layout="wide")

# --- simple CSS for blue + white theme
st.markdown(
    """
    <style>
    :root {
        --accent: #2B9AF3;
        --muted: #f5f7fb;
        --card: #ffffff;
        --text: #212529;
    }
    .stApp { background-color: var(--muted); }
    header { background: white; box-shadow: 0 1px 0 rgba(0,0,0,0.04); }
    .sidebar .sidebar-content { background: #f8fbff; }
    .title-row { display:flex; align-items:center; gap:12px; }
    .big-title { font-size:30px; font-weight:700; color:var(--text); }
    .accent { color: var(--accent); }
    .metric-card { background: var(--card); padding: 18px; border-radius: 8px; border: 1px solid #e6eefc; }
    .small-muted { color: #6c757d; font-size:14px; }
    </style>
    """, unsafe_allow_html=True
)

# --------------------------
# Helpers: caching loaders
# --------------------------
@st.cache_resource(show_spinner=False)
def load_pickle(path):
    """Load a pickle if available, otherwise return None and exception."""
    if not os.path.exists(path):
        return None, FileNotFoundError(f"{path} not found")
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj, None
    except Exception as e:
        return None, e

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None, FileNotFoundError(f"{path} not found")
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, e

# --------------------------
# Paths (repo layout)
# --------------------------
# app/app.py expects:
# ../models/logistic_model.pkl
# ../models/random_forest_model.pkl
# ../models/scaler.pkl
# ../data/diabetes.csv
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

LOGISTIC_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")
RF_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CSV_PATH = os.path.join(DATA_DIR, "diabetes.csv")

# --------------------------
# Load models & data
# --------------------------
with st.sidebar.expander("Model & Data status", expanded=True):
    st.write("sklearn version on the server:", sklearn.__version__)

log_reg, err1 = load_pickle(LOGISTIC_PATH)
rf, err2 = load_pickle(RF_PATH)
scaler, err3 = load_pickle(SCALER_PATH)
df, err4 = load_csv(CSV_PATH)

# show diagnostics if any
diagnostics = []
if err1:
    diagnostics.append(f"logistic_model: {repr(err1)}")
if err2:
    diagnostics.append(f"random_forest_model: {repr(err2)}")
if err3:
    diagnostics.append(f"scaler: {repr(err3)}")
if err4:
    diagnostics.append(f"data csv: {repr(err4)}")

if diagnostics:
    with st.sidebar:
        st.warning("One or more required files have issues. See details:")
        for d in diagnostics:
            st.text(d)
        st.markdown("---")
        st.info("If pickles were created with an older sklearn, re-save them with the current environment or retrain locally and re-upload.")
else:
    with st.sidebar:
        st.success("All models & data loaded successfully ✅")
        st.text("Model classes:")
        try:
            st.text(f"Logistic: {type(log_reg)}")
            st.text(f"RandomForest: {type(rf)}")
            st.text(f"Scaler: {type(scaler)}")
        except Exception:
            pass

# --------------------------
# Sidebar navigation & inputs
# --------------------------
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Go to:", ["Prediction Dashboard", "Dataset Explorer", "Model Insights"])

st.sidebar.markdown("---")
st.sidebar.markdown("#### Created by Mayowa Oluyole")
st.sidebar.markdown("📧 oluyolemayowa1@gmail.com")
st.sidebar.markdown("---")

# default input values (safe)
default_inputs = {
    "Pregnancies": 0,
    "Glucose": 120.0,
    "BloodPressure": 70.0,
    "SkinThickness": 20.0,
    "Insulin": 80.0,
    "BMI": 30.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 33
}

# --------------------------
# PAGE: Prediction Dashboard
# --------------------------
if page == "Prediction Dashboard":
    # Header
    st.markdown('<div class="title-row"><span class="big-title">🩺 Diabetes Prediction Dashboard</span></div>', unsafe_allow_html=True)
    st.write("Enter patient clinical measurements and predict diabetes risk using two models.")

    # Input column (left)
    with st.container():
        col1, col2 = st.columns([1, 3])

        with col1:
            st.header("Patient Inputs")
            preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=default_inputs["Pregnancies"], step=1)
            glucose = st.number_input("Glucose", min_value=0.0, max_value=400.0, value=default_inputs["Glucose"])
            bp = st.number_input("Blood Pressure", min_value=0.0, max_value=250.0, value=default_inputs["BloodPressure"])
            skin = st.number_input("Skin Thickness", min_value=0.0, max_value=200.0, value=default_inputs["SkinThickness"])
            insulin = st.number_input("Insulin", min_value=0.0, max_value=2000.0, value=default_inputs["Insulin"])
            bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=default_inputs["BMI"])
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=default_inputs["DiabetesPedigreeFunction"])
            age = st.number_input("Age", min_value=1, max_value=120, value=default_inputs["Age"])

            st.markdown("---")
            st.markdown("**Model & Environment**")
            st.write(f"sklearn: {sklearn.__version__}")
            if log_reg is not None:
                st.write(f"Logistic model type: {getattr(log_reg, '__class__', 'unknown')}")
            if rf is not None:
                st.write(f"RandomForest model type: {getattr(rf, '__class__', 'unknown')}")

        with col2:
            # display input summary table
            input_data = pd.DataFrame([{
                "Pregnancies": preg,
                "Glucose": glucose,
                "BloodPressure": bp,
                "SkinThickness": skin,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age
            }])
            st.subheader("🔎 Input Summary")
            st.dataframe(input_data, use_container_width=True)

            # Prediction action
            if st.button("Predict Diabetes Risk"):
                if any([err1, err2, err3]):
                    st.error("Cannot run prediction because models or scaler failed to load. Check the sidebar diagnostics.")
                else:
                    try:
                        X_scaled = scaler.transform(input_data)
                        lr_pred = log_reg.predict(X_scaled)[0]
                        lr_prob = float(log_reg.predict_proba(X_scaled)[0][1]) if hasattr(log_reg, "predict_proba") else float(0.0)
                        rf_pred = rf.predict(X_scaled)[0]
                        rf_prob = float(rf.predict_proba(X_scaled)[0][1]) if hasattr(rf, "predict_proba") else float(0.0)

                        # Show results
                        left, right = st.columns(2)
                        with left:
                            st.markdown("### Logistic Regression")
                            st.write("Prediction:", "🟥 Diabetes" if lr_pred == 1 else "🟩 No Diabetes")
                            st.metric("Probability", f"{lr_prob:.2f}")
                        with right:
                            st.markdown("### Random Forest")
                            st.write("Prediction:", "🟥 Diabetes" if rf_pred == 1 else "🟩 No Diabetes")
                            st.metric("Probability", f"{rf_prob:.2f}")

                        # Interpretation
                        st.subheader("📌 Risk Interpretation (Random Forest)")
                        if rf_prob > 0.7:
                            st.error("High Risk — Medical evaluation recommended immediately.")
                        elif rf_prob > 0.4:
                            st.warning("Moderate Risk — Monitor closely.")
                        else:
                            st.success("Low Risk — Maintain healthy lifestyle.")
                    except Exception as e:
                        st.exception(f"Error during prediction: {e}")

# --------------------------
# PAGE: Dataset Explorer
# --------------------------
elif page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")
    if err4:
        st.error("Dataset failed to load. Check sidebar diagnostics.")
    else:
        st.subheader("🔹 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("🔹 Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("🔹 Correlation Heatmap")
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax1)
        ax1.set_title("Correlation matrix")
        st.pyplot(fig1)

        st.subheader("🔹 Glucose Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.histplot(df['Glucose'].dropna(), kde=True, ax=ax2)
        ax2.set_title("Glucose distribution")
        st.pyplot(fig2)

# --------------------------
# PAGE: Model Insights
# --------------------------
elif page == "Model Insights":
    st.title("🤖 Model Insights & Interpretability")

    if any([err1, err2, err3, err4]):
        st.error("Some models/data failed to load. See sidebar diagnostics.")
    else:
        # Prepare dataset
        df_local = df.copy()
        missing_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for c in missing_zero_cols:
            df_local[c] = df_local[c].replace(0, df_local[c].median())

        X = df_local.drop(columns=["Outcome"], errors="ignore")
        y = df_local["Outcome"] if "Outcome" in df_local.columns else pd.Series([])

        # Scale features for models (use scaler already provided)
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            st.exception(f"Error scaling dataset: {e}")
            X_scaled = None

        # Show basic metrics
        if X_scaled is not None and len(y) > 0:
            X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
            )

            def metrics(y_true, pred):
                return (
                    skm.accuracy_score(y_true, pred),
                    skm.precision_score(y_true, pred, zero_division=0),
                    skm.recall_score(y_true, pred, zero_division=0)
                )

            try:
                lr_acc, lr_prec, lr_rec = metrics(y_test_m, log_reg.predict(X_test_m))
            except Exception:
                lr_acc = lr_prec = lr_rec = 0.0
            try:
                rf_acc, rf_prec, rf_rec = metrics(y_test_m, rf.predict(X_test_m))
            except Exception:
                rf_acc = rf_prec = rf_rec = 0.0

            st.subheader("🔹 Model Performance (Test Set)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Logistic Regression")
                st.metric("Accuracy", f"{lr_acc:.3f}")
                st.metric("Precision", f"{lr_prec:.3f}")
                st.metric("Recall", f"{lr_rec:.3f}")
            with c2:
                st.markdown("### Random Forest")
                st.metric("Accuracy", f"{rf_acc:.3f}")
                st.metric("Precision", f"{rf_prec:.3f}")
                st.metric("Recall", f"{rf_rec:.3f}")

            st.markdown("---")
            # Feature importance (RandomForest)
            try:
                st.subheader("🔹 Feature Importance (Random Forest)")
                importances = pd.Series(rf.feature_importances_, index=X.columns)
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                importances.sort_values().plot(kind="barh", ax=ax3)
                ax3.set_title("Feature importance (Random Forest)")
                st.pyplot(fig3)
            except Exception as e:
                st.info("Could not compute RF feature importances: " + str(e))

            st.markdown("---")
            # Permutation importance
            try:
                st.subheader("🔹 Permutation Importance (Model Sensitivity)")
                perm_result = permutation_importance(rf, X_test_m, y_test_m, n_repeats=10, random_state=42)
                perm_importance = pd.Series(perm_result.importances_mean, index=X.columns)
                fig4, ax4 = plt.subplots(figsize=(8, 6))
                perm_importance.sort_values().plot(kind="barh", ax=ax4)
                ax4.set_title("Permutation importance (Random Forest)")
                st.pyplot(fig4)
            except Exception as e:
                st.info("Could not compute permutation importance: " + str(e))

            st.markdown("---")
            # Partial dependence for top feature
            try:
                st.subheader("🔹 Partial Dependence Plot (Top RF Feature)")
                top_feature = importances.sort_values(ascending=False).index[0]
                st.write(f"Partial dependence for **{top_feature}**")
                fig5 = plt.figure(figsize=(7, 5))
                PartialDependenceDisplay.from_estimator(rf, X, [top_feature], ax=plt.gca())
                st.pyplot(fig5)
            except Exception as e:
                st.info("Could not compute partial dependence: " + str(e))
        else:
            st.info("Not enough data to run model insights. Make sure the dataset contains an 'Outcome' column and models loaded.")

# --------------------------
# Footer
# --------------------------
st.markdown("""<br><hr><div style='text-align:center;font-size:14px;color:#8892a6'>
Created by <b>Mayowa Oluyole</b> • Diabetes Prediction ML Dashboard • © 2025
</div>""", unsafe_allow_html=True)
