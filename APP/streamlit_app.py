import streamlit as st
import torch
import numpy as np
import os
from model import model
from utils import load_csv, load_hdf5, preprocess_ecg_data

# Streamlit app
st.title('ECG Prediction App')
st.header('Detection of silent heart attacks')


uploaded_file = st.file_uploader("Choose an ECG file", type=["csv", "hdf5"])
file_type = st.selectbox("Select File Type", ["csv", "hdf5"])
dataset_name = st.text_input("Dataset Name (for HDF5)", value="ecg_dataset")
target_length = st.number_input("Target Length", value=5300)

if st.button("Predict"):
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join('/tmp', uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            if file_type == 'csv':
                ecg_data = load_csv(temp_file_path)
            elif file_type == 'hdf5':
                ecg_data = load_hdf5(temp_file_path, dataset_name)
            else:
                st.error("Unsupported file type")
                st.stop()
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
        finally:
            os.remove(temp_file_path)  # Clean up temporary file
        try:
            ecg_data = preprocess_ecg_data(ecg_data, target_length)
        except ValueError as e:
            st.error(f"Error preprocessing data: {e}")
            st.stop()

        ecg_data_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)  # Shape (1, 12, target_length)
        with torch.no_grad():
            output = model(ecg_data_tensor)
            prediction = torch.sigmoid(output).item()  # Get scalar prediction

        st.success(f"Prediction: {prediction}")
    else:
        st.error("No file uploaded")
