import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("water_model.pkl")

st.title("💧 Water Potability Prediction App")

st.write("Enter the water quality parameters below:")

# Text inputs (typing method)
ph = st.text_input("pH value")
hardness = st.text_input("Hardness")
solids = st.text_input("Solids")
chloramines = st.text_input("Chloramines")
sulfate = st.text_input("Sulfate")
conductivity = st.text_input("Conductivity")
organic_carbon = st.text_input("Organic Carbon")
trihalomethanes = st.text_input("Trihalomethanes")
turbidity = st.text_input("Turbidity")

if st.button("Predict Potability"):
    try:
        # Convert all to float
        values = [float(ph), float(hardness), float(solids), float(chloramines),
                  float(sulfate), float(conductivity), float(organic_carbon),
                  float(trihalomethanes), float(turbidity)]

        # Make prediction
        prediction = model.predict([values])

        # Display result
        if prediction[0] == 1:
            st.success("💧 The water is POTABLE (Safe to drink)")
        else:
            st.error("⚠️ The water is NOT POTABLE (Unsafe to drink)")

    except ValueError:
        st.warning("⚠️ Please enter valid numeric values in all fields!")
