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
with open('models/logistic_model.pkl', 'rb') as f:
    log_reg = pickle.load(f)

with open('models/random_forest_model.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Prediction Dashboard", "Dataset Explorer", "Model Insights"]
)

# --------------------------------------------------
# PAGE 1: PREDICTION DASHBOARD
# --------------------------------------------------
if page == "Prediction Dashboard":
    st.title("🩺 Diabetes Prediction Dashboard")
    st.write("Enter the patient's clinical measurements to predict diabetes risk.")

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

# --------------------------------------------------
# PAGE 2: DATASET EXPLORER
# --------------------------------------------------
elif page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.dataframe(df.head())

# --------------------------------------------------
# PAGE 3: MODEL INSIGHTS
# --------------------------------------------------
elif page == "Model Insights":
    st.title("🤖 Model Insights & Interpretability")

    df_local = df.copy()
    missing_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for c in missing_cols:
        df_local[c] = df_local[c].replace(0, df_local[c].median())

    X = df_local.drop('Outcome', axis=1)
    y = df_local['Outcome']
    X_scaled = scaler.transform(X)

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    lr_acc = skm.accuracy_score(y_test_m, log_reg.predict(X_test_m))
    rf_acc = skm.accuracy_score(y_test_m, rf.predict(X_test_m))

    st.metric("Logistic Regression Accuracy", f"{lr_acc:.3f}")
    st.metric("Random Forest Accuracy", f"{rf_acc:.3f}")
