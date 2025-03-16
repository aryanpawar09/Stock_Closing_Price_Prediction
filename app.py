import streamlit as st
import tensorflow as tf
import numpy as np
import pickle  # Use pickle instead of joblib

# Load the trained GRU model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("gru_stock_model.keras")
    return model

model = load_model()

# Load the scaler using pickle
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("📈 Stock Price Prediction using GRU")
st.write("Enter the stock data to predict the next closing price.")

# Define the number of input features and time steps
n_features = 2  # Update this if your model uses a different number of features
time_steps = 10  # Update this to match the time steps used in training

# Input fields for stock features
input_data = []
st.write(f"Enter the last {time_steps} values for each feature:")

for i in range(time_steps):
    row = []
    for j in range(n_features):
        row.append(st.number_input(f"Feature {j+1}, Time Step {i+1}", value=0.0))
    input_data.append(row)

# Convert input data to a NumPy array
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, time_steps, n_features)  # Shape: (1, time_steps, n_features)

    # Scale the input data using the same scaler from training
    scaled_input = scaler.transform(input_array.reshape(-1, n_features)).reshape(1, time_steps, n_features)

    # Make prediction
    prediction = model.predict(scaled_input)
    
    # Display prediction
    st.success(f"Predicted Closing Price: {prediction[0][0]:.4f}")

    # Debugging: Print input details
    st.write("Debug Info:")
    st.write(f"Original Input Shape: {np.array(input_data).shape}")
    st.write(f"Scaled Input Shape: {scaled_input.shape}")
    st.write(f"Model Input Shape: {input_array.shape}")
