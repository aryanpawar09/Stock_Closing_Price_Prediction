import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import requests
import os

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("gru_stock_model.keras")
    return model

# Download and load the scaler from GitHub
@st.cache_resource
def load_scaler():
    scaler_url = "https://raw.githubusercontent.com/aryanpawar09/Stock_Closing_Price_Prediction/main/scaler.pkl"
    response = requests.get(scaler_url)

    if response.status_code == 200:
        with open("scaler.pkl", "wb") as f:
            f.write(response.content)
        return joblib.load("scaler.pkl")
    else:
        st.error("Failed to download scaler.pkl. Please check the URL.")
        return None

model = load_model()
scaler = load_scaler()

st.title("Stock Closing Price Prediction")

# User input for stock features
feature1 = st.number_input("Enter Feature 1:")
feature2 = st.number_input("Enter Feature 2:")

if st.button("Predict"):
    if scaler:
        input_data = np.array([[feature1, feature2]])
        input_scaled = scaler.transform(input_data).reshape(1, 1, 2)

        prediction = model.predict(input_scaled)
        st.success(f"Predicted Closing Price: {prediction[0][0]:.2f}")
    else:
        st.error("Scaler not loaded. Check the scaler file.")

