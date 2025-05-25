# app.py

import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('engine_condition_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Engine Condition Predictor", layout="centered")

st.title("🚗 Engine Condition Predictor")
st.markdown("Enter the engine parameters to predict whether the engine condition is **Normal (0)** or **Faulty (1)**.")

# Input fields
engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=10000, value=800)
lub_oil_pressure = st.number_input("Lub Oil Pressure", value=3.0)
fuel_pressure = st.number_input("Fuel Pressure", value=12.0)
coolant_pressure = st.number_input("Coolant Pressure", value=3.5)
lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", value=85.0)
coolant_temp = st.number_input("Coolant Temperature (°C)", value=82.0)

# Predict button
if st.button("Predict Engine Condition"):
    input_data = np.array([[engine_rpm, lub_oil_pressure, fuel_pressure,
                            coolant_pressure, lub_oil_temp, coolant_temp]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = "🟢 Normal (0)" if prediction == 0 else "🔴 Faulty (1)"
    st.success(f"Prediction: **{result}**")
