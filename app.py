import streamlit as st
import pickle
import numpy as np

# Load both models
port_model = pickle.load(open('water_model.pkl', 'rb'))
use_model = pickle.load(open('water_use_model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

st.title("💧 Water Quality & Usage Prediction App")
st.write("This AI-powered app predicts whether water is potable and what it is best suitable for — Agriculture 🌾, Industry 🏭, or Drinking 🚰")

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

if st.button("Predict"):
    features = np.array([[pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])

    # Predictions
    potable_pred = port_model.predict(features)[0]
    use_pred = use_model.predict(features)[0]
    usage_label = label_encoder.inverse_transform([use_pred])[0]

    # Conditional logic to fix mismatch
    if potable_pred == 0:
        potable_result = "⚠️ The water is NOT POTABLE (Unsafe to drink)."
        # Override usage suggestion if unsafe
        suggested_use = "🚫 Not recommended for Drinking. Use for Agriculture 🌾 or Industrial 🏭 purposes only."
    else:
        potable_result = "✅ The water is POTABLE (Safe for drinking)."
        suggested_use = f"💧 Suggested Usage: {usage_label}"

    # Display results
    st.subheader("🔹 Prediction Results")
    st.write(potable_result)
    st.write(suggested_use)

