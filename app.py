import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("gru_stock_model.keras")
    return model

model = load_model()

# Streamlit UI
st.title("📈 Stock Price Prediction using GRU")
st.write("Enter the stock data to predict the next closing price.")

# Input fields for stock features
n_features = 2  # Change based on your dataset
input_data = []
for i in range(n_features):
    input_data.append(st.number_input(f"Feature {i+1}", value=0.0))

# Convert input to numpy array and reshape for model
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, 1, n_features)  # Ensure correct shape
    prediction = model.predict(input_array)
    
    st.success(f"Predicted Closing Price: {prediction[0][0]:.4f}")
