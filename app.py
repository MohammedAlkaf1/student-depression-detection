
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Smart Campus Mental Health Detector",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('label_encoders.pkl')
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

st.title("🧠 Student Depression Risk Detector")
st.caption("A Smart Campus Mental Health Early Detection System")
st.markdown("""
This application uses machine learning to identify students who may be at risk 
of depression based on academic and lifestyle factors. Built for **BCI3333 - 
Machine Learning Applications**.
""")
st.divider()

with st.sidebar:
    st.header("ℹ️ About")
    st.write("**Model:** Logistic Regression")
    st.write("**Accuracy:** 84.25%")
    st.write("**Recall:** 88.00%")
    st.write("**Dataset:** 27,901 student records")
    st.divider()
    st.warning("⚠️ This tool is for educational purposes only. Not a clinical diagnosis.")

st.subheader("📋 Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 60, 22)
    academic_pressure = st.slider("Academic Pressure (1-5)", 1.0, 5.0, 3.0, 0.5)
   cgpa_input = st.slider("CGPA (Malaysian scale 0-4)", 0.0, 4.0, 3.0, 0.01)
    cgpa = cgpa_input * 2.5
    study_satisfaction = st.slider("Study Satisfaction (1-5)", 1.0, 5.0, 3.0, 0.5)

with col2:
    sleep_duration = st.selectbox("Sleep Duration",
                                  ["Less than 5 hours", "5-6 hours",
                                   "7-8 hours", "More than 8 hours"])
    dietary_habits = st.selectbox("Dietary Habits",
                                  ["Healthy", "Moderate", "Unhealthy"])
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?",
                                     ["No", "Yes"])
    work_study_hours = st.slider("Work/Study Hours per day", 0.0, 12.0, 6.0, 0.5)
    financial_stress = st.slider("Financial Stress (1-5)", 1.0, 5.0, 3.0, 0.5)

family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

st.divider()

if st.button("🔍 Predict Mental Health Status", type="primary", use_container_width=True):
    input_dict = {
        'Gender': encoders['Gender'].transform([gender])[0],
        'Age': age,
        'Academic Pressure': academic_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_satisfaction,
        'Sleep Duration': encoders['Sleep Duration'].transform([sleep_duration])[0],
        'Dietary Habits': encoders['Dietary Habits'].transform([dietary_habits])[0],
        'Have you ever had suicidal thoughts ?':
            encoders['Have you ever had suicidal thoughts ?'].transform([suicidal_thoughts])[0],
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness':
            encoders['Family History of Mental Illness'].transform([family_history])[0]
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **At Risk of Depression**")
        st.write(f"Risk probability: **{probability[1]*100:.1f}%**")
        st.info("""
        **Recommendation:** This student shows patterns associated with depression risk.
        - Visit campus counselling services
        - Talk to a trusted friend, family member, or mentor
        - Consider reaching out to a mental health professional
        """)
    else:
        st.success(f"✅ **Low Risk of Depression**")
        st.write(f"Confidence: **{probability[0]*100:.1f}%**")
        st.info("""
        **Recommendation:** Current patterns suggest low depression risk.
        Continue maintaining healthy habits:
        - Regular sleep schedule
        - Balanced diet
        - Manage academic workload
        - Stay socially connected
        """)

    st.write("**Probability Breakdown:**")
    prob_df = pd.DataFrame({
        'Status': ['No Depression', 'Depression'],
        'Probability': [f"{probability[0]*100:.2f}%", f"{probability[1]*100:.2f}%"]
    })
    st.dataframe(prob_df, hide_index=True, use_container_width=True)

st.divider()
st.caption("Developed for BCI3333 Machine Learning Applications - Universiti Malaysia Pahang Al-Sultan Abdullah")
