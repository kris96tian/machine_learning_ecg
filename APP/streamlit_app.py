import streamlit as st
import requests
import torch
import numpy as np
import pandas as pd
import h5py
from io import BytesIO
from model import ECGCNN, load_model_from_dropbox  # Make sure this matches your model file

st.set_page_config(page_title="ECG Prediction App", layout="centered")

st.title("ECG Prediction App")
st.subheader("Detection of Silent Heart Attacks")

uploaded_file = st.file_uploader("Choose ECG file", type=["csv", "hdf5"])
file_type = st.selectbox("Select File Type", options=["csv", "hdf5"])
dataset_name = st.text_input("Dataset Name (for HDF5)", placeholder="(Optional, for HDF5 files)")
target_length = st.number_input("Target Length", min_value=1, value=5300)

# Load the model
dropbox_url = 'https://www.dropbox.com/scl/fi/f0nehqdr5i2salflap0gg/model.pth?rlkey=p38eox3ci5w8ky1abhc0uz2cx&st=nmcqw6cb&dl=1'
model = load_model_from_dropbox(dropbox_url)

if model is None:
    st.error("Failed to load the model.")
else:
    model.eval()

    if st.button("Predict"):
        if uploaded_file is not None:
            try:
                # Load and process the uploaded file
                ecg_data = load_data(uploaded_file, file_type, dataset_name, target_length)
                
                # Perform prediction
                ecg_data_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                with torch.no_grad():
                    prediction = model(ecg_data_tensor).item()

                # Display prediction result
                st.success(f"Prediction: {prediction:.4f}")
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

# Utility functions
def load_csv(file_path):
    ecg_data = pd.read_csv(file_path).values
    return ecg_data.T  # -> to (leads, time_points)

def load_hdf5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        ecg_data = f[dataset_name][:]
    return ecg_data.T  # -> to (leads, time_points)

def preprocess_ecg_data(ecg_data, target_length=5300):
    num_leads, num_points = ecg_data.shape
    
    if num_leads != 12:
        raise ValueError(f"Expected 12 leads, but got {num_leads} leads.")
    
    # Adjusting data to have exactly `target_length` time points
    if num_points > target_length:
        ecg_data = ecg_data[:, :target_length] 
    elif num_points < target_length:
        padding = target_length - num_points
        ecg_data = np.pad(ecg_data, ((0, 0), (0, padding)), 'constant')  
    
    return ecg_data

def load_data(uploaded_file, file_type, dataset_name, target_length):
    if file_type == "csv":
        ecg_data = load_csv(uploaded_file)
    elif file_type == "hdf5":
        ecg_data = load_hdf5(uploaded_file, dataset_name)
    else:
        raise ValueError("Unsupported file type")
    
    return preprocess_ecg_data(ecg_data, target_length)
