import streamlit as st
import joblib
import numpy as np

# Load models
potability_model = joblib.load("water_model.pkl")
use_model = joblib.load("water_use_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("💧 Water Quality & Usage Prediction App")

st.write("""
This AI-powered app predicts:
1. Whether the water is **Potable (Safe for Drinking)**  
2. What the water is **Best Suitable For** → Agriculture 🌾 | Industry 🏭 | Drinking 🚰
""")

# Input fields
ph = st.text_input("pH value")
hardness = st.text_input("Hardness")
solids = st.text_input("Solids")
chloramines = st.text_input("Chloramines")
sulfate = st.text_input("Sulfate")
conductivity = st.text_input("Conductivity")
organic_carbon = st.text_input("Organic Carbon")
trihalomethanes = st.text_input("Trihalomethanes")
turbidity = st.text_input("Turbidity")

if st.button("🔍 Predict Water Quality"):
    try:
        # Convert inputs to float
        values = [float(ph), float(hardness), float(solids), float(chloramines),
                  float(sulfate), float(conductivity), float(organic_carbon),
                  float(trihalomethanes), float(turbidity)]

        X = np.array([values])

        # Predict potability
        potable_pred = potability_model.predict(X)[0]

        # Predict usage type
        use_pred = use_model.predict(X)
        use_label = label_encoder.inverse_transform(use_pred)[0]

        # Display results
        st.subheader("🔹 Prediction Results")

        if potable_pred == 1:
            st.success("✅ The water is POTABLE (Safe to drink).")
            st.info("💧 Suggested Usage: **Drinking Water 🚰**")
        else:
            st.error("⚠️ The water is NOT POTABLE (Unsafe to drink).")

            # Show only Agriculture or Industry based on model prediction
            if use_label.lower() == "agriculture":
               st.info("💧 Suggested Usage: **Agriculture Water 🌾**")
            elif use_label.lower() == "industry":
               st.info("💧 Suggested Usage: **Industrial Water 🏭**")
            else:
               st.warning("💧 Suggested Usage: **Not suitable for any common use**")

           
    except ValueError:
        st.warning("⚠️ Please enter valid numeric values in all fields!")
