import streamlit as st
import torch
import numpy as np
import pandas as pd
import h5py
from torch import nn
import os
import io

# Model definition
class ECGCNN(nn.Module):
    def __init__(self):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 662, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Utility functions
def load_csv(file):
    df = pd.read_csv(file)
    return df.values.T

def load_hdf5(file, dataset_name):
    with h5py.File(file, 'r') as f:
        ecg_data = f[dataset_name][:]
    return ecg_data.T

def preprocess_ecg_data(ecg_data, target_length=5300):
    num_leads, num_points = ecg_data.shape
    
    if num_leads != 12:
        raise ValueError(f"Expected 12 leads, but got {num_leads} leads.")
    
    if num_points > target_length:
        ecg_data = ecg_data[:, :target_length]
    elif num_points < target_length:
        padding = target_length - num_points
        ecg_data = np.pad(ecg_data, ((0, 0), (0, padding)), 'constant')
    
    return ecg_data

def load_model():
    model = ECGCNN()
    try:
        if not os.path.exists('model.ph'):
            st.error("Model file not found. Please ensure 'model.ph' is in the same directory as this script.")
            return None
            
        # Load with weights_only=True for security
        checkpoint = torch.load(
            'model.ph', 
            map_location=torch.device('cpu'),
            weights_only=True  # More secure loading
        )
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint)
        else:
            state_dict = checkpoint
            
        # Filter and load state dict
        model_state_dict = model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        
        # Check if we have any valid weights
        if not filtered_state_dict:
            st.error("No valid weights found in model file.")
            return None
            
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()  # Set to evaluation mode
        return model
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


# Streamlit UI
def main():
    st.title("ECG Diagnose Prediction App (Silent Heart Attacks)")
    st.markdown("""
    ### Prediction made upon your ECG-Data (uploaded as CSV or HDF5 format) by using a pre-trained Deep-Learning Model.
    """)

    # File upload
    file_type = st.selectbox("Select file type", ["CSV", "HDF5"])
    uploaded_file = st.file_uploader(f"Upload your ECG file ({file_type} format)", type=[file_type.lower()])

    # Dataset name input for HDF5
    dataset_name = None
    if file_type == "HDF5":
        dataset_name = st.text_input("Dataset name (for HDF5 files)")

    # Target length input
    target_length = st.number_input("Target length of ECG data", value=5300, min_value=1)

    # Sample data download button
    if st.button("Download sample CSV"):
        # You'll need to provide the sample data
        sample_data = pd.DataFrame(np.random.randn(5300, 12))  # Replace with actual sample data
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download sample CSV file",
            data=csv,
            file_name="sampledata.csv",
            mime="text/csv"
        )

    if uploaded_file is not None:
        try:
            if file_type == "CSV":
                ecg_data = load_csv(uploaded_file)
            else:  # HDF5
                if not dataset_name:
                    st.error("Please provide a dataset name for the HDF5 file")
                    return
                ecg_data = load_hdf5(uploaded_file, dataset_name)

            ecg_data = preprocess_ecg_data(ecg_data, target_length)

            model = load_model()
            if model is not None:
                model.eval()
                with torch.no_grad():
                    ecg_data_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)
                    output = model(ecg_data_tensor)
                    prediction = torch.sigmoid(output).item()
                    prediction_percentage = prediction * 100

                st.success(f"Prediction: {prediction_percentage:.2f}%")

                st.line_chart(pd.DataFrame(ecg_data.T))

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    st.markdown("""
    **Note:** The model was trained on data from 3,750 patients, with each ECG sample having a shape of (5300, 12), 
    representing 5300 data points across 12 ECG leads.
    """)

if __name__ == "__main__":
    main()
