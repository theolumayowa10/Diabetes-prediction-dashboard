import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


# --------------------------------------------------
# LOAD MODELS & SCALER
# --------------------------------------------------
with open('../models/logistic_model.pkl', 'rb') as f:
    log_reg = pickle.load(f)

with open('../models/random_forest_model.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset
df = pd.read_csv("../data/diabetes.csv")


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS (UI POLISH)
# --------------------------------------------------
st.markdown("""
<style>
body { background-color: #f8f9fc; }
.sidebar .sidebar-content { background-color: #eef2f5; }
h1, h2, h3, h4 { color: #2c3e50; }
.footer { text-align: center; font-size: 14px; color: gray; padding-top: 10px; }
.metric-card { background-color: #ffffff; padding: 18px; border-radius: 8px; border: 1px solid #e3e6ea; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Prediction Dashboard", "Dataset Explorer", "Model Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Created by Mayowa Oluyole")
st.sidebar.markdown("📧 oluyolemayowa1@gmail.com")
st.sidebar.markdown("---")


# --------------------------------------------------
# PAGE 1: PREDICTION DASHBOARD
# --------------------------------------------------
if page == "Prediction Dashboard":

    st.title("🩺 Diabetes Prediction Dashboard")
    st.write("Enter the patient's clinical measurements to predict diabetes risk.")

    st.sidebar.header("Patient Information")

    preg = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.number_input("Glucose Level", 0.0, 300.0, 120.0)
    bp = st.sidebar.number_input("Blood Pressure", 0.0, 200.0, 70.0)
    skin = st.sidebar.number_input("Skin Thickness", 0.0, 100.0, 20.0)
    insulin = st.sidebar.number_input("Insulin Level", 0.0, 900.0, 80.0)
    bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 30.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.number_input("Age", 10, 120, 33)

    input_data = pd.DataFrame({
        "Pregnancies": [preg],
        "Glucose": [glucose],
        "BloodPressure": [bp],
        "SkinThickness": [skin],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    st.subheader("🔎 Input Summary")
    st.dataframe(input_data)

    if st.button("Predict Diabetes Risk"):
        scaled_data = scaler.transform(input_data)

        lr_pred = log_reg.predict(scaled_data)[0]
        lr_prob = log_reg.predict_proba(scaled_data)[0][1]

        rf_pred = rf.predict(scaled_data)[0]
        rf_prob = rf.predict_proba(scaled_data)[0][1]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Logistic Regression")
            st.write("Prediction:", "🟥 Diabetes" if lr_pred == 1 else "🟩 No Diabetes")
            st.metric("Probability", f"{lr_prob:.2f}")

        with col2:
            st.markdown("### Random Forest")
            st.write("Prediction:", "🟥 Diabetes" if rf_pred == 1 else "🟩 No Diabetes")
            st.metric("Probability", f"{rf_prob:.2f}")

        st.subheader("📌 Risk Interpretation (Random Forest)")
        if rf_prob > 0.7:
            st.error("High Risk — Medical evaluation recommended immediately.")
        elif rf_prob > 0.4:
            st.warning("Moderate Risk — Monitor closely.")
        else:
            st.success("Low Risk — Maintain healthy lifestyle.")


# --------------------------------------------------
# PAGE 2: DATASET EXPLORER
# --------------------------------------------------
elif page == "Dataset Explorer":

    st.title("📊 Dataset Explorer")

    st.subheader("🔹 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("🔹 Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("🔹 Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
    st.pyplot(fig1)

    st.subheader("🔹 Glucose Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Glucose'], kde=True, color="blue", ax=ax2)
    st.pyplot(fig2)


# --------------------------------------------------
# PAGE 3: MODEL INSIGHTS
# --------------------------------------------------
elif page == "Model Insights":

    st.title("🤖 Model Insights & Interpretability")

    df_local = df.copy()
    missing_zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for c in missing_zero_cols:
        df_local[c] = df_local[c].replace(0, df_local[c].median())

    X = df_local.drop('Outcome', axis=1)
    y = df_local['Outcome']
    X_scaled = scaler.transform(X)

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # METRICS
    def metrics(y_true, pred):
        return (
            skm.accuracy_score(y_true, pred),
            skm.precision_score(y_true, pred, zero_division=0),
            skm.recall_score(y_true, pred, zero_division=0)
        )

    lr_acc, lr_prec, lr_rec = metrics(y_test_m, log_reg.predict(X_test_m))
    rf_acc, rf_prec, rf_rec = metrics(y_test_m, rf.predict(X_test_m))

    st.subheader("🔹 Model Performance (Test Set)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Logistic Regression")
        st.metric("Accuracy", f"{lr_acc:.3f}")
        st.metric("Precision", f"{lr_prec:.3f}")
        st.metric("Recall", f"{lr_rec:.3f}")

    with col2:
        st.markdown("### Random Forest")
        st.metric("Accuracy", f"{rf_acc:.3f}")
        st.metric("Precision", f"{rf_prec:.3f}")
        st.metric("Recall", f"{rf_rec:.3f}")

    st.markdown("---")

    # FEATURE IMPORTANCE
    st.subheader("🔹 Feature Importance (Random Forest)")
    importances = pd.Series(rf.feature_importances_, index=X.columns)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(kind='barh', ax=ax3)
    ax3.set_title("Feature Importance")
    st.pyplot(fig3)

    st.markdown("---")

    # PERMUTATION IMPORTANCE
    st.subheader("🔹 Permutation Importance (Model Sensitivity)")

    perm_result = permutation_importance(
        rf, X_test_m, y_test_m, n_repeats=10, random_state=42
    )

    perm_importance = pd.Series(
        perm_result.importances_mean,
        index=X.columns
    )

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    perm_importance.sort_values().plot(kind='barh', ax=ax4, color="teal")
    ax4.set_title("Permutation Importance (Random Forest)")
    st.pyplot(fig4)

    st.markdown("---")

    # PARTIAL DEPENDENCE PLOT
    st.subheader("🔹 Partial Dependence Plot (Feature Impact Curve)")
    top_feature = importances.sort_values(ascending=False).index[0]

    st.write(f"Partial Dependence for **{top_feature}**")

    fig5 = plt.figure(figsize=(7, 5))
    PartialDependenceDisplay.from_estimator(
        rf, X, [top_feature], ax=plt.gca()
    )
    st.pyplot(fig5)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<br><br>
<hr>
<div class="footer">
Created by <b>Mayowa Oluyole</b> • Diabetes Prediction ML Dashboard • © 2025
</div>
""", unsafe_allow_html=True)
