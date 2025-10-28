import streamlit as st
import joblib
import numpy as np
import os

# Load models safely
st.title("💧 Water Quality & Usage Prediction App")
st.write("This AI-powered app predicts whether water is potable and what it is best suitable for — Agriculture 🌾, Industry 🏭, or Drinking 🚰")

try:
    port_model = joblib.load('water_model.pkl')
    use_model = joblib.load(open('water_use_model.pkl','rb'))
    label_encoder = joblib.load(open('label_encoder.pkl','rb'))
    model_loaded = True
except Exception as e:
    st.error("⚠️ Error loading models. Please check if all .pkl files are uploaded correctly.")
    st.stop()

# Input fields
pH = st.number_input("pH value")
Hardness = st.number_input("Hardness")
Solids = st.number_input("Solids")
Chloramines = st.number_input("Chloramines")
Sulfate = st.number_input("Sulfate")
Conductivity = st.number_input("Conductivity")
Organic_carbon = st.number_input("Organic Carbon")
Trihalomethanes = st.number_input("Trihalomethanes")
Turbidity = st.number_input("Turbidity")

# Prediction button
if st.button("Predict"):
    features = np.array([[pH, Hardness, Solids, Chloramines, Sulfate, Conductivity,
                          Organic_carbon, Trihalomethanes, Turbidity]])

    # Model predictions
    potable_pred = port_model.predict(features)[0]
    use_pred = use_model.predict(features)[0]
    usage_label = label_encoder.inverse_transform([use_pred])[0]

    # Display results
    st.subheader("🔹 Prediction Results")
    if potable_pred == 0:
        st.error("⚠️ The water is NOT POTABLE (Unsafe to drink).")
        st.warning("🚫 Suggested Usage: Not recommended for Drinking. Suitable for Agriculture 🌾 or Industrial 🏭 use only.")
    else:
        st.success("✅ The water is POTABLE (Safe for drinking).")
        st.info(f"💧 Suggested Usage: {usage_label}")

