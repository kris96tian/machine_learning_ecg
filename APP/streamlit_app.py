import streamlit as st
import requests
import os
import numpy as np
import h5py
import pandas as pd
from model import model  # Ensure `predict_ecg` is defined in `model`
from utils import load_csv, load_hdf5, preprocess_ecg_data

st.set_page_config(page_title="ECG Prediction App", layout="centered")

st.title("ECG Prediction App")
st.subheader("Detection of Silent Heart Attacks")

uploaded_file = st.file_uploader("Choose ECG file", type=["csv", "hdf5"])

file_type = st.selectbox("Select File Type", options=["csv", "hdf5"])

dataset_name = st.text_input("Dataset Name (for HDF5)", placeholder="(Optional, for HDF5 files)")

target_length = st.number_input("Target Length", min_value=1, value=5300)

def load_data(uploaded_file, file_type, dataset_name, target_length):
    # Convert uploaded file to a file-like object
    file_path = uploaded_file
    
    # Load the data based on the file type
    if file_type == 'csv':
        ecg_data = load_csv(file_path)
    elif file_type == 'hdf5':
        ecg_data = load_hdf5(file_path, dataset_name)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or HDF5 file.")
    
    # Preprocess the data
    ecg_data = preprocess_ecg_data(ecg_data, target_length)
    
    return ecg_data

if st.button("Predict"):
    if uploaded_file is not None:
        try:
            # Load and process the uploaded file
            ecg_data = load_data(uploaded_file, file_type, dataset_name, target_length)
            
            # Perform prediction
            prediction = predict_ecg(ecg_data)  # Ensure `predict_ecg` is defined
            
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
