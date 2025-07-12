import streamlit as st
import pandas as pd
import joblib
import os

model = joblib.load(os.path.join("model", "rf_hypertension_model.pkl"))

st.title("Hypertension Risk Predictor")

age = st.number_input("Age", 10, 100)
male = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
currentSmoker = st.selectbox("Smoker? (0=No, 1=Yes)", [0, 1])
cigsPerDay = st.number_input("Cigarettes/day ", 0, 60)
BPMeds = st.selectbox("On BP meds? (0=No, 1=Yes)", [0, 1])
diabetes = st.selectbox("Diabetes? (0=No, 1=Yes)", [0, 1])
totChol = st.number_input("Total Cholesterol", 100, 400)
sysBP = st.number_input("Systolic BP", 80, 200)
diaBP = st.number_input("Diastolic BP", 50, 130)
BMI = st.number_input("BMI", 10.0, 50.0)
heartRate = st.number_input("Heart Rate", 40, 200)
glucose = st.number_input("Glucose", 60, 250)

if st.button("Predict Hypertension Risk"):
    input_data = pd.DataFrame([[
        male, age, currentSmoker, cigsPerDay, BPMeds, diabetes,
        totChol, sysBP, diaBP, BMI, heartRate, glucose
    ]], columns=[
        "male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes",
        "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
    ])
    result = model.predict(input_data)[0]
    if result == 1:
        st.error("High Risk of Hypertension")
    else:
        st.success("Low Risk of Hypertension")
