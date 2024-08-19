import streamlit as st
import torch
import numpy as np
import os
from model import model
from utils import load_csv, load_hdf5, preprocess_ecg_data

# Define constants
TEMP_DIR = '/tmp'  # Directory to store temporary files

def save_temp_file(uploaded_file):
    """Save the uploaded file to a temporary directory."""
    temp_file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return temp_file_path

def predict_ecg(file_path, file_type, dataset_name, target_length):
    """Load, preprocess, and predict ECG data."""
    # Load ECG data based on file type
    if file_type == 'csv':
        ecg_data = load_csv(file_path)
    elif file_type == 'hdf5':
        ecg_data = load_hdf5(file_path, dataset_name)
    else:
        raise ValueError("Unsupported file type")

    # Preprocess ECG data
    ecg_data = preprocess_ecg_data(ecg_data, target_length)

    # Convert data to tensor and make prediction
    ecg_data_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(ecg_data_tensor)
        prediction = torch.sigmoid(output).item()

    # Convert prediction to percentage
    prediction_percentage = prediction * 100
    return prediction_percentage

def main():
    st.title("ECG Signal Prediction")

    st.write("Upload your ECG file and choose the appropriate settings to get a prediction.")

    # File upload widget
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'hdf5'])

    # Input widgets for file type, dataset name, and target length
    file_type = st.selectbox("Select file type", ['csv', 'hdf5'])
    dataset_name = st.text_input("Dataset name (only for HDF5 files)", value="")
    target_length = st.number_input("Target length of ECG data", min_value=100, max_value=10000, value=5300)

    if uploaded_file is not None:
        try:
            # Save the uploaded file temporarily
            file_path = save_temp_file(uploaded_file)

            # Show progress bar
            with st.spinner('Processing...'):
                prediction_percentage = predict_ecg(file_path, file_type, dataset_name, target_length)
            
            st.success(f"Prediction: {prediction_percentage:.2f}%")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == '__main__':
    main()
