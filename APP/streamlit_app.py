import streamlit as st
import requests
import os

from model import model 
from utils import load_csv,  load_hdf5, preprocess_ecg_data

st.set_page_config(page_title="ECG Prediction App", layout="centered")

st.title("ECG Prediction App")
st.subheader("Detection of Silent Heart Attacks")

uploaded_file = st.file_uploader("Choose ECG file", type=["csv", "hdf5"])

file_type = st.selectbox("Select File Type", options=["csv", "hdf5"])

dataset_name = st.text_input("Dataset Name (for HDF5)", placeholder="(Optional, for HDF5 files)")

target_length = st.number_input("Target Length", min_value=1, value=5300)

if st.button("Predict"):
    if uploaded_file is not None:
        try:
            # Load and process the uploaded file
            ecg_data = load_data(uploaded_file, file_type, dataset_name, target_length)
            
            # Perform prediction
            prediction = predict_ecg(ecg_data)

            # Display prediction result
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please upload an ECG file.")

# Footer
st.markdown("""
    <style>
        .footer {
            background-color: #050505;
            color: #fff;
            text-align: center;
            padding: 15px 0;
            width: 100%;
            position: fixed;
            bottom: 0;
            left: 0;
        }
    </style>
    <div class="footer">
        <p>Created by Kristian Alikaj. For more visit <a href="https://github.com/kris96tian" target="_blank">GitHub profile</a>.</p>
    </div>
    """, unsafe_allow_html=True)
