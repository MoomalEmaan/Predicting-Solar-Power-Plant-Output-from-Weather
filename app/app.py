import json
from pathlib import Path
import numpy as np
import streamlit as st

# Load the Set B weights and scaling statistics saved by train_eval.py
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "results" / "weights.json"
with open(WEIGHTS_PATH) as f:
    model = json.load(f)["B"]
mean = np.array(model["mean"])
std = np.array(model["std"])
theta = np.array(model["theta"])

def predict(hour, sw_radiation, temp_2m, cloud_cover):
    # Same feature order as training: sw_radiation, temp_2m, cloud_cover, sin_hour, cos_hour
    x = np.array([
        sw_radiation / 1000,  # model was trained on kW/m2
        temp_2m,
        cloud_cover,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
    ])
    x_scaled = (x - mean) / std
    x_with_intercept = np.concatenate(([1.0], x_scaled))
    # Clip at zero, a plant cannot produce negative power
    return max(float(x_with_intercept @ theta), 0.0)

st.title("Solar Plant 1 - AC Power Predictor")
st.write("Predicts hourly AC power of Plant 1 from public weather data only "
         "(Set B, normal-equation weights).")

hour = st.slider("Hour of day", 0, 23, 12)
sw_radiation = st.number_input("Shortwave radiation (W/m²)", min_value=0.0, max_value=1200.0, value=600.0, step=10.0)
temp_2m = st.number_input("Air temperature at 2 m (°C)", min_value=-10.0, max_value=55.0, value=33.0, step=0.5)
cloud_cover = st.slider("Cloud cover (%)", 0, 100, 50)

if st.button("Predict"):
    power = predict(hour, sw_radiation, temp_2m, cloud_cover)
    st.metric("Predicted AC power", f"{power:,.0f} kW")
