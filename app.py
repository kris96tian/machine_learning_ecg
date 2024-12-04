import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import h5py
import streamlit as st

# Function to load CSV data
def load_csv(file_path):
    ecg_data = pd.read_csv(file_path).values
    return ecg_data.T  # Convert to (leads, time_points)

# Function to load HDF5 data
def load_hdf5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        ecg_data = f[dataset_name][:]
    return ecg_data.T  # Convert to (leads, time_points)

# Preprocessing function to handle ECG data shape
def preprocess_ecg_data(ecg_data, target_length=5300):
    num_leads, num_points = ecg_data.shape
    
    # Ensure the data has 12 leads
    if num_leads != 12:
        raise ValueError(f"Expected 12 leads, but got {num_leads} leads.")
    
    # Adjust data length to match target length
    if num_points > target_length:
        ecg_data = ecg_data[:, :target_length]  # Truncate if too long
    elif num_points < target_length:
        padding = target_length - num_points
        ecg_data = np.pad(ecg_data, ((0, 0), (0, padding)), 'constant')  # Pad if too short
    
    return ecg_data


# Define the ECGCNN model
class ECGCNN(nn.Module):
    def __init__(self):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 662, 128)  # 128 * 662 = 84736
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer
        return x


# Function to load a trained model
def load_model(model_path='model.ph'):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = ECGCNN()  # Initialize the model

    checkpoint_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model_state_dict = model.state_dict()
    
    # Filter weights to ensure compatibility
    filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}

    model.load_state_dict(filtered_state_dict, strict=False)  # Load the weights into the model
    return model


# Load and preprocess ECG data
def prepare_data(file_path, dataset_name=None, file_type='csv'):
    if file_type == 'csv':
        ecg_data = load_csv(file_path)
    elif file_type == 'hdf5':
        ecg_data = load_hdf5(file_path, dataset_name)
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'hdf5'.")

    return preprocess_ecg_data(ecg_data)


# Define the Streamlit interface
def main():
    st.title('ECG Prediction using Deep Learning')

    # File uploader for ECG data
    uploaded_file = st.file_uploader("Choose an ECG file", type=["csv", "h5"])
    if uploaded_file is not None:
        # Handle the uploaded file based on its extension
        file_type = uploaded_file.name.split('.')[-1]
        if file_type == 'csv':
            ecg_data = prepare_data(uploaded_file)
        elif file_type == 'h5':
            ecg_data = prepare_data(uploaded_file, dataset_name='ecg_dataset', file_type='hdf5')
        else:
            st.error("Invalid file format. Please upload a CSV or HDF5 file.")
            return

        # Load the trained model
        model = load_model('model.ph')

        # Prepare the data for input to the model
        ecg_data_tensor = torch.tensor(ecg_data).unsqueeze(0).float()  # Add batch dimension

        # Perform inference
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = model(ecg_data_tensor)  # Make predictions

        # Show prediction result
        st.write(f'Predicted output: {output.item()}')

# Run the app
if __name__ == "__main__":
    main()
