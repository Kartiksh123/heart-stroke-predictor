import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Heart Stroke Predictor", layout="centered")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
h1 {
    text-align: center;
    color: #ff4b4b;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #ff2b2b;
}
.card {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown("<h1>❤️ Heart Stroke Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-based prediction using Machine Learning</p>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    columns = joblib.load("heart_columns.pkl")
    return model, scaler, columns

model, scaler, expected_columns = load_model()

# ------------------ INPUT SECTION ------------------
st.subheader("🧾 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)

with col2:
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", [
        "Atypical Angina", "Non-Anginal Pain",
        "Typical Angina", "Asymptomatic"
    ])
    fasting_bs = st.radio("Fasting Blood Sugar > 120", ["Yes", "No"])
    fasting_bs = 1 if fasting_bs == "Yes" else 0

    exercise_angina = st.radio("Exercise Angina", ["Yes", "No"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("---")

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict Risk"):

    with st.spinner("Analyzing data... ⏳"):

        input_dict = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_Normal': 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([input_dict])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled_input)[0][1]
            st.markdown(f"<h3 style='text-align:center;'>📊 Risk Score: {prob:.2f}</h3>", unsafe_allow_html=True)

        # RESULT UI
        if prediction == 1:
            st.markdown("""
            <div class='card' style='background-color:#ffcccc;'>
                <h2 style='color:red;'>⚠️ High Risk of Heart Disease</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='card' style='background-color:#ccffcc;'>
                <h2 style='color:green;'>✅ Low Risk of Heart Disease</h2>
            </div>
            """, unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Made with ❤️ by Kartik Sharma</p>",
    unsafe_allow_html=True
)