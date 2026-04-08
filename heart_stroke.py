import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = "knn_heart_model.pkl"
scaler_path = "heart_scaler.pkl"
columns_path = "heart_columns.pkl"
print("MODEL:", model_path)
print("SCALER:", scaler_path)
print("COLUMNS:", columns_path)
# Page config
st.set_page_config(page_title="Heart Stroke Predictor", layout="centered")

# Load model files (must be in same folder)
try:
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title
st.title("❤️ Heart Stroke Prediction by Kartik")

st.markdown("""
This application estimates your risk of heart disease based on health metrics.
⚠️ *For educational purposes only. Not medical advice.*
""")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("ML model: KNN | Built using Streamlit")

# Inputs
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])

chest_pain_type_options = [
    "Atypical Angina",
    "Non-Anginal Pain",
    "Typical Angina",
    "Asymptomatic"
]
chest_pain = st.selectbox("Chest Pain Type", chest_pain_type_options)

resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL?", ["Yes", "No"])
fasting_bs = 1 if fasting_bs == "Yes" else 0

resting_ecg_options = [
    "Normal",
    "ST-T Wave Abnormality",
    "Left Ventricular Hypertrophy"
]
resting_ecg = st.selectbox("Resting ECG", resting_ecg_options)

max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)

exercise_angina = st.radio("Exercise-induced angina?", ["Yes", "No"])

oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Predict
if st.button("Predict"):

    try:
        # Input dictionary
        input_dict = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,

            # One-hot encoding
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Add missing columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure order
        input_df = input_df[expected_columns]

        # Scale
        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_input)[0]

        # Probability (if available)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled_input)[0][1]
            st.write(f"📊 Risk Score: {prob:.2f}")

        # Result
        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")