import streamlit as st
import numpy as np
import torch
from model import ECGCNN  
from utils import load_csv, load_hdf5, preprocess_ecg_data

def load_model_from_dropbox(url):
    try:
        import requests
        from io import BytesIO
        response = requests.get(url)
        response.raise_for_status()  
        buffer = BytesIO(response.content)
        model = ECGCNN()
        model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
        model.eval()
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
dropbox_url = 'https://www.dropbox.com/scl/fi/f0nehqdr5i2salflap0gg/model.pth?rlkey=p38eox3ci5w8ky1abhc0uz2cx&st=nmcqw6cb&dl=1'
model = load_model_from_dropbox(dropbox_url)

def load_data(uploaded_file, file_type, dataset_name, target_length):
    if file_type == 'csv':
        ecg_data = load_csv(uploaded_file)
    elif file_type == 'hdf5':
        ecg_data = load_hdf5(uploaded_file, dataset_name)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or HDF5 file.")
    ecg_data = preprocess_ecg_data(ecg_data, target_length)
    return ecg_data

def predict_ecg(ecg_data):
    if model is None:
        raise RuntimeError("Model not loaded.")
    ecg_data = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(ecg_data)
    return output.item()



# Streamlit 
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
            ecg_data = load_data(uploaded_file, file_type, dataset_name, target_length)
            #prediction
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
