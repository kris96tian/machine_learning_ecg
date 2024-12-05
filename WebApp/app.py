import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import io
import gradio as gr

# Model
class ECGCNN(nn.Module):
    def __init__(self):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 662, 128)
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

# Utility fncs
def load_csv(file_content: bytes) -> np.ndarray:
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        return df.values.T
    except Exception as e:
        raise ValueError(f"Error processing CSV file: {str(e)}")
def load_hdf5(file_content: bytes, dataset_name: str) -> np.ndarray:
    try:
        with h5py.File(io.BytesIO(file_content), 'r') as f:
            return f[dataset_name][:].T
    except Exception as e:
        raise ValueError(f"Error processing HDF5 file: {str(e)}")

def preprocess_ecg_data(ecg_data: np.ndarray, target_length: int = 5300) -> np.ndarray:
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
    checkpoint = torch.load('model.ph', map_location=torch.device('cpu'), weights_only=True)  
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint)
    else:
        state_dict = checkpoint
    model_state = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def predict_ecg(file, file_type, dataset_name=None):
    global model
    if 'model' not in globals():
        model = load_model()

    if hasattr(file, 'read'):
        file_content = file.read()
    elif hasattr(file, 'name'):
        with open(file.name, 'rb') as f:
            file_content = f.read()
    else:
        file_content = file  
    
    if file_type.lower() == "csv":
        ecg_data = load_csv(file_content)
    elif file_type.lower() == "hdf5":
        if not dataset_name:
            raise ValueError("Dataset name is required for HDF5 files")
        ecg_data = load_hdf5(file_content, dataset_name)
    else:
        raise ValueError("Unsupported file type. Only CSV and HDF5 are supported.")

    # Preprocess 
    ecg_data = preprocess_ecg_data(ecg_data)
    
    #  prediction
    with torch.no_grad():
        ecg_data_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)
        output = model(ecg_data_tensor)
        prediction = torch.sigmoid(output).item()
        prediction_percentage = prediction * 100
    
    #  risk level
    if prediction_percentage > 75:
        risk_level = "High Risk"
    elif prediction_percentage > 25:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    return f"{prediction_percentage:.2f}%", risk_level

def create_gradio_interface():
    gr.HTML('<h1>ECG Diagnose Prediction App<br>(Silent Heart Attacks)</h1>')
    gr.Markdown("""
        ### Prediction made upon your ECG-Data (uploaded as CSV or HDF5 format) by using a pre-trained Deep-Learning Model.
        
        **Note:** The model was trained on data from 3,750 patients, with each ECG sample having a shape of (5300, 12), representing 5300 data points across 12 ECG leads.
        """)
    
    # Interface definition
    iface = gr.Interface(
        fn=predict_ecg,
        inputs=[
            gr.File(label="Upload ECG File"),
            gr.Dropdown(["CSV", "HDF5"], label="File Type"),
            gr.Textbox(label="Dataset Name (for HDF5, optional)", value="")
        ],
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Textbox(label="Risk Level")
        ],
        title="ECG Heart Attack Risk Prediction",
        description="Upload an ECG file to predict heart attack risk"
    )

    # Footer 
    gr.Markdown("""
        ### Created by Kristian Alikaj
        [GitHub Repository](https://github.com/kris96tian/machine_learning_ecg)
        """)

    return iface

# Main execution
if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch()
