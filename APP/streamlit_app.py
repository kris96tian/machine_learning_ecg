import streamlit as st
import torch
import numpy as np
import os
from model import model
from utils import load_csv, load_hdf5, preprocess_ecg_data

# Streamlit app
st.title('ECG Diagnose Prediction App (Silent Heart Attacks) ')

st.markdown("""
This app allows you to analyze ECG data using a pre-trained AI model. Follow these steps to use the tool:
1. Select your ECG file using the "Choose ECG file" button. Supported formats include CSV and HDF5 for now.
2. Choose the file type from the dropdown menu. (in case of a HDF5 file, you can specify the dataset name)
3. Set the desired target length for your data. The tool will automatically trim or pad your ECG data to match this length.
5. Click "Predict" to analyze the ECG data. The prediction result will be displayed below.

**Note:** The model was trained on data from 3,750 patients, with each ECG sample having a shape of (5300, 12), representing 5300 data points across 12 ECG leads.
""")
# File upload
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

        # Load ECG data based on file type
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

        # Preprocess ECG data
        try:
            ecg_data = preprocess_ecg_data(ecg_data, target_length)
        except ValueError as e:
            st.error(f"Error preprocessing data: {e}")
            st.stop()

        # Convert data to tensor and make prediction
        ecg_data_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)  # Shape (1, 12, target_length)
        with torch.no_grad():
            output = model(ecg_data_tensor)
            prediction = torch.sigmoid(output).item()  # Get scalar prediction
            prediction_percentage = prediction * 100


        # Display prediction
        st.success(f"Prediction for future silent heart attack:  {prediction_percentage:.2f}%")
    else:
        st.error("No file uploaded")

# footer
st.markdown("""
Created by Kristian Alikaj using PyTorch and Streamlit. You can find the source code on [GitHub repository](https://github.com/kris96tian/machine_learning_ecg) """ )
