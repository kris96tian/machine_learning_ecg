import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import io
import gradio as gr

# Model Definition
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

# Utility Functions
def load_csv(file_content: bytes) -> np.ndarray:
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        return df.values.T
    except Exception as e:
        raise ValueError(f"Error processing CSV file: {str(e)}")

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
    checkpoint = torch.load('model.ph', map_location=torch.device('cpu'))
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Predict Function
def predict_ecg(file, file_type):
    global model
    if 'model' not in globals():
        model = load_model()

    file_content = file.read()
    ecg_data = load_csv(file_content)
    ecg_data = preprocess_ecg_data(ecg_data)

    with torch.no_grad():
        ecg_data_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)
        output = model(ecg_data_tensor)
        prediction = torch.sigmoid(output).item() * 100  # Convert to percentage

    risk_level = (
        "High Risk" if prediction > 75
        else "Medium Risk" if prediction > 25
        else "Low Risk"
    )
    return f"{prediction:.2f}%", risk_level

def download_example_data():
    with open("example_data.csv", "rb") as f:
        return f.read()

# Gradio Interface
iface = gr.Blocks()

with iface:
    gr.Markdown("# ECG Heart Attack Risk Prediction")
    gr.Markdown(
        """
        Upload an ECG file to predict heart attack risk. 
        - **File Format**: CSV with 12 leads (columns) and 5300 timepoints (rows).  
        - Or click below to download an example CSV file.
        """
    )
    with gr.Row():
        download_button = gr.File(label="Download Example Data", file_types=["csv"], value="example_data.csv")
    file = gr.File(label="Upload ECG File")
    file_type = gr.Dropdown(["CSV"], label="File Type")
    predict_button = gr.Button("Predict")
    prediction = gr.Textbox(label="Prediction")
    risk_level = gr.Textbox(label="Risk Level")

    predict_button.click(
        predict_ecg,
        inputs=[file, file_type],
        outputs=[prediction, risk_level],
    )

iface.launch(server_name="0.0.0.0", server_port=7860)
