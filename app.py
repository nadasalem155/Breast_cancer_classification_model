import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set up page configuration: title, icon, and layout
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load pre-trained decision tree model and scaler
model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature information dictionary:
# key: feature name
# value: (emoji, tooltip description with 'Why it matters' explanation)
features_info = {
    'concavity_worst': (
        "🔻",
        "Measures the deepest inward curves in the tumor's outline (worst case). Irregular shapes with deep concavities are more common in malignant tumors.\nWhy it matters: Irregularities here can indicate malignancy."
    ),
    'radius_mean': (
        "📏",
        "Average distance from tumor center to edge. Larger tumors may require more attention and indicate higher risk.\nWhy it matters: Larger radius often correlates with more advanced tumors."
    ),
    'concave_points_mean': (
        "⭕️",
        "Average number of concave points in the tumor outline. More concave points suggest irregular shapes, often linked to malignancy.\nWhy it matters: More concave points usually imply tumor irregularity."
    ),
    'area_se': (
        "📐",
        "Standard error of the tumor's area measurement. High variation can indicate irregular or changing tumor shape.\nWhy it matters: High variation signals unstable or uneven tumor growth."
    ),
    'radius_worst': (
        "📏🔥",
        "Largest measured radius of the tumor. Larger maximum size can be a sign of advanced tumor growth.\nWhy it matters: Largest radius reflects potential tumor progression."
    ),
    'perimeter_worst': (
        "🌀",
        "Largest perimeter measured around the tumor. A longer perimeter combined with irregularity is often concerning.\nWhy it matters: Complex perimeters often correlate with malignancy."
    ),
    'concave_points_worst': (
        "⭕️🔥",
        "Highest number of concave points measured. High count of concave points indicates irregular, potentially malignant growth.\nWhy it matters: High concave points indicate tumor aggressiveness."
    )
}

# Add CSS to increase spacing between input columns
st.markdown("""
    <style>
        /* Increase horizontal gap between columns */
        .css-1lcbmhc.e1fqkh3o3 {
            gap: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# Display title and subtitle centered on the page
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🔬 Breast Cancer Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter the values for the 7 features to check if the tumor is malignant or benign.</p>", unsafe_allow_html=True)
st.write("---")

# Create three columns for input fields
user_input = []
cols = st.columns(3)

for i, (feature, (emoji, tooltip)) in enumerate(features_info.items()):
    with cols[i % 3]:
        # Compose label with emoji and feature name
        label = f"{emoji} {feature.replace('_', ' ').capitalize()}"
        
        # Create a number input with a tooltip (help text shown on hover)
        value = st.number_input(
            label,
            value=0.0,
            format="%.4f",
            help=tooltip,
            label_visibility="visible"
        )
        user_input.append(value)

st.write("")

# When the Predict button is clicked
if st.button("🔍 Predict", use_container_width=True):
    # Create DataFrame from user inputs with correct feature columns
    input_df = pd.DataFrame([user_input], columns=features_info.keys())

    # Scale inputs using the pre-loaded scaler
    scaled_input = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_input, columns=features_info.keys())

    # Make prediction using the loaded model
    prediction = model.predict(scaled_df)[0]

    # Display result with styled message
    if prediction == 1:
        st.markdown(
            """
            <div style="background-color: #FFB3B3; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #B80000;">🚨 Malignant</h2>
                <p style="font-size: 18px;">The tumor is likely malignant. Please consult a doctor for further diagnosis.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color: #B3FFCC; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #006400;">✅ Benign</h2>
                <p style="font-size: 18px; color: black;">The tumor is likely benign. Regular check-ups are still recommended.</p>
            </div>
            """,
            unsafe_allow_html=True
        )