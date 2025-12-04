import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from joblib import load

df = pd.read_csv("data/raw/diabetes.csv")

model = load("models/xgboost_bayesian.joblib")
scaler = load("models/scaler.joblib")


st.set_page_config(page_title="Diabetes Prediction", page_icon="🤓", layout="centered")

st.title("Diabetes Progression Prediction App")
st.write(
    "This app predicts the **disease progression after 1 year (Y)** using patient medical features."
)

st.divider()

st.header("Dataset Quick Summary")


st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Feature Summary")
st.write(df.describe().T)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(df.corr(), cmap="coolwarm")
st.pyplot(fig)

st.subheader("First 10 rows")
st.dataframe(df.head(10))

st.divider()

st.header("Enter Patient Data to Predict Y")

features = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]
feature_labels = {
    "AGE": "Age (in years)",
    "SEX": "Sex (1 = Male, 2 = Female)",
    "BMI": "BMI (Body Mass Index)",
    "BP": "Mean Arterial Blood Pressure",
    "S1": "S1 – Total Cholesterol Level (TC)",
    "S2": "S2 – LDL Cholesterol (Bad Cholesterol)",
    "S3": "S3 – HDL Cholesterol (Good Cholesterol)",
    "S4": "S4 – Cholesterol Ratio (TC/HDL)",
    "S5": "S5 – Log Triglycerides (LTG)",
    "S6": "S6 – Blood Glucose Level",
}


user_input = {}

cols = st.columns(2)

for i, feat in enumerate(features):
    label = feature_labels[feat]

    with cols[i % 2]:
        if feat == "AGE":
            user_input[feat] = st.number_input(
                label, min_value=0, max_value=120, value=int(df[feat].mean()), step=1
            )

        if feat == "SEX":
            user_input[feat] = st.selectbox(label, [1, 2])

        else:
            user_input[feat] = st.number_input(
                label, value=float(df[feat].mean()), format="%.4f"
            )


user_df = pd.DataFrame([user_input])

st.write("### Your Input Data")
st.dataframe(user_df)

st.divider()

st.header("Prediction Result")

if st.button("Predict Y"):
    user_scaled = scaler.transform(user_df)

    y_pred = model.predict(user_scaled)[0]

    st.success(f"### Predicted Diabetes Progression (Y): **{y_pred:.2f}**")
