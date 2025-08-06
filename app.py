import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the feature names
feature_names = [
    'concavity_worst', 'radius_mean', 'concave_points_mean',
    'area_se', 'radius_worst', 'perimeter_worst', 'concave_points_worst'
]

st.title("Breast Cancer Prediction (Decision Tree)")
st.markdown("Enter the values for the 7 features to predict if the tumor is malignant or benign.")

# Create input fields
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature.replace('_', ' ').capitalize()}", value=0.0, format="%.4f")
    user_input.append(value)

if st.button("Predict"):
    # Prepare input as DataFrame with correct column names
    input_df = pd.DataFrame([user_input], columns=feature_names)

    # Scale the input
    scaled_input = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_input, columns=feature_names)

    # Predict
    prediction = model.predict(scaled_df)[0]

    # Show result
    if prediction == 1:
        st.success("🔬 The tumor is Malignant.")
    else:
        st.info("🩺 The tumor is Benign.")