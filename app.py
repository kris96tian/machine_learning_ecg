import streamlit as st
import torch
import numpy as np
import os
import tempfile
from model import ECGCNN
from utils import load_csv, load_hdf5, preprocess_ecg_data
import pickle 

# Page config
st.set_page_config(
    page_title="ECG Diagnose Prediction App",
    page_icon="❤️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0D1B2A;
        color: #E0E1DD;
    }
    .stButton>button {
        background-color: #4b155b;
        color: white;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #4f99c4;
    }
</style>
""", unsafe_allow_html=True)


def load_model(model_path='model.ph'):
    try:
        # Attempt to load the checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

        print("Model checkpoint loaded successfully.")
        
        # Initialize the model (code for model initialization goes here)
        return model
    except FileNotFoundError:
        print(f"Error: The model file '{model_path}' was not found.")
    except pickle.UnpicklingError as e:
        print(f"Error: UnpicklingError occurred while loading the model: {e}")
    except Exception as e:
        print(f"Error loading model: {e}")


# Main app
def main():
    st.title("ECG Diagnose Prediction App\n(Silent Heart Attacks)")
    st.subheader("Prediction made upon your ECG-Data using a pre-trained Deep-Learning Model")

    model = load_model()

    # File upload
    uploaded_file = st.file_uploader("Upload your ECG file", type=['csv', 'hdf5'])
    file_type = st.selectbox("Select file type", ['csv', 'hdf5'])
    
    # Optional inputs
    dataset_name = None
    if file_type == 'hdf5':
        dataset_name = st.text_input("Dataset name (only for HDF5 files)")
    
    target_length = st.number_input("Target length of ECG data", 
                                  min_value=100, 
                                  max_value=10000, 
                                  value=5300)

    if st.button("PREDICT"):
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name

                # Load and process data
                if file_type == 'csv':
                    ecg_data = load_csv(temp_file_path)
                elif file_type == 'hdf5':
                    if not dataset_name:
                        st.error("Please provide a dataset name for HDF5 files.")
                        return
                    ecg_data = load_hdf5(temp_file_path, dataset_name)

                ecg_data = preprocess_ecg_data(ecg_data, target_length)
                
                # Make prediction
                ecg_data_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    output = model(ecg_data_tensor)
                    prediction = torch.sigmoid(output).item()
                    prediction_percentage = prediction * 100

                # Display results
                if prediction > 0.5:
                    st.success(f"Prediction: Heart Attack detected with {prediction_percentage:.2f}% certainty.")
                else:
                    st.success(f"Prediction: Normal ECG with {prediction_percentage:.2f}% certainty.")

                # Cleanup
                os.unlink(temp_file_path)

            except FileNotFoundError:
                st.error("The file was not found. Please check the file path.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.error("Please upload a file")

    st.markdown("""
    **Note:** The model was trained on data from 3,750 patients, with each ECG sample having 
    a shape of (5300, 12), representing 5300 data points across 12 ECG leads.
    """)

    st.markdown("""
    ---
    Created by Kristian Alikaj using Streamlit and PyTorch. 
    [GitHub repository](https://github.com/kris96tian/machine_learning_ecg)
    """)

if __name__ == '__main__':
    main()
