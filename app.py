import streamlit as st
import pickle
import joblib
import numpy as np

# -------------------------------
# Load model, features, and scaler
# -------------------------------
with open("best_model_rf_top10.pkl", "rb") as f:
    model = pickle.load(f)

with open("important_features_top10.pkl", "rb") as f:
    feature_list = pickle.load(f)

scaler = joblib.load("num_scaler.pkl")  # Scaler trained on all 10 features

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="NYC Taxi Fare Predictor", layout="centered")
st.title("🚕 NYC Taxi Fare Predictor")
st.markdown("Predict the total fare based on trip details (Top 10 features).")

# -------------------------------
# Input Feature Widgets
# -------------------------------
ratecode_ids = [1, 2, 3, 4, 5, 6]
zones = list(range(1, 264))
weekdays = list(range(0, 7))  # 0 = Monday
hours = list(range(0, 24))

# Initialize dictionary to hold user input
user_inputs = {}

# Group numeric inputs into 2 columns
col1, col2 = st.columns(2)

with col1:
    user_inputs["trip_distance"] = st.number_input("Trip Distance (miles)", min_value=0.0, value=31.5)
    user_inputs["tip_amount"] = st.number_input("Tip Amount ($)", min_value=0.0, value=2.3)
    user_inputs["mta_tax"] = st.number_input("MTA Tax ($)", min_value=0.0, value=0.6, step=0.1)
    user_inputs["RatecodeID"] = st.selectbox("Ratecode ID", ratecode_ids)

with col2:
    user_inputs["trip_duration"] = st.number_input("Trip Duration (minutes)", min_value=0.0, value=18.0)
    user_inputs["passenger_count"] = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
    user_inputs["hour"] = st.selectbox("Pickup Hour", hours)
    user_inputs["weekday"] = st.selectbox("Day of Week (0=Mon)", weekdays)

# Group zone dropdowns in a horizontal row
col3, col4 = st.columns(2)
with col3:
    user_inputs["PUZone"] = st.selectbox("Pickup Zone", zones)
with col4:
    user_inputs["DOZone"] = st.selectbox("Drop-off Zone", zones)

# -------------------------------
# Prepare input data in correct feature order
# -------------------------------
raw_input = np.array([user_inputs[feature] for feature in feature_list]).reshape(1, -1)

# Apply scaler to the full input (all 10 features)
scaled_input = scaler.transform(raw_input)

# -------------------------------
# Predict and Display
# -------------------------------
if st.button("Predict Fare"):
    predicted_fare = model.predict(scaled_input)[0]
    st.success(f"💰 Estimated Total Fare: **${predicted_fare:.2f}**")

# -------------------------------
# Footer
# -------------------------------
st.markdown(""" 
     <div style='text-align: center; font-size: 0.8em; color: gray;'>Developed by Harsh Bang (<a href='https://www.linkedin.com/in/harshbang/' target='_blank'>LinkedIn Profile</a>)
    </div>
    """, unsafe_allow_html=True)
