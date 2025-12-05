import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from joblib import load


@st.cache_data
def load_dataset():
    return pd.read_csv("data/raw/diabetes.csv")


@st.cache_resource
def load_model():
    model = load("models/elasticnet_iqr.joblib")
    scaler = load("models/scaler.joblib")
    return model, scaler


def render_dataset_summary(df):
    st.header("Dataset Quick Summary")

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Feature Summary")
    st.write(df.describe().T)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("First 10 rows")
    st.dataframe(df.head(10))


def render_input_form(df):
    st.header("Enter Patient Data to Predict Y")

    feats = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]
    lbls = {
        "AGE": "Age (years)",
        "SEX": "Sex",
        "BMI": "BMI (Body Mass Index)",
        "BP": "Mean Arterial Blood Pressure",
        "S1": "S1 – Total Cholesterol (TC)",
        "S2": "S2 – LDL Cholesterol (Bad)",
        "S3": "S3 – HDL Cholesterol (Good)",
        "S4": "S4 – Cholesterol Ratio (TC/HDL)",
        "S5": "S5 – Log Triglycerides",
        "S6": "S6 – Blood Glucose Level",
    }

    user_input = {}
    cols = st.columns(2)

    for i, feat in enumerate(feats):
        with cols[i % 2]:
            if feat == "AGE":
                user_input[feat] = st.number_input(
                    lbls[feat],
                    min_value=0,
                    max_value=120,
                    value=int(df[feat].mean()),
                    step=1,
                )

            elif feat == "SEX":
                user_input[feat] = (
                    1 if st.radio(lbls[feat], ["Male", "Female"]) == "Male" else 2
                )

            else:
                user_input[feat] = st.number_input(
                    lbls[feat],
                    value=float(df[feat].mean()),
                    format="%.4f",
                )

    user_df = pd.DataFrame([user_input])

    st.write("### Your Input Data")
    st.dataframe(user_df)

    return user_df


def predict_value(model, scaler, user_df):
    st.header("Prediction Result")

    if st.button("Predict Y"):
        user_scaled = scaler.transform(user_df)
        y_pred = model.predict(user_scaled)[0]

        st.success(f"### Predicted Diabetes Progression (Y): **{y_pred:.2f}**")


def main():
    st.set_page_config(
        page_title="Diabetes Prediction",
        page_icon="🤓",
        layout="centered",
    )

    st.title("Diabetes Progression Prediction App")
    st.write(
        "This app predicts the **disease progression after 1 year (Y)** using patient medical features."
    )

    st.divider()

    df = load_dataset()
    model, scaler = load_model()

    render_dataset_summary(df)

    st.divider()

    user_df = render_input_form(df)

    st.divider()

    predict_value(model, scaler, user_df)


if __name__ == "__main__":
    main()
