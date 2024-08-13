import torch
import torch.nn as nn
from mega import Mega
import tempfile
import os

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
        
        x = x.view(x.size(0), -1)  
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model_from_mega(url):
    try:     
        mega = Mega()
        m = mega.login()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Download the file to the temporary location
        m.download_url(url, temp_path)
        
        model = ECGCNN()
        checkpoint = torch.load(temp_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
        # Delete the temporary file
        os.unlink(temp_path)
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Ensure temporary file is deleted even if an error occurs
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        return None

# Load the model
mega_url = 'https://mega.nz/file/eMQyUbTL#rN13DEjWCpqp0RUYoA2fVyl4xeaErlbkj8z3WGI7gUg'
model = load_model_from_mega(mega_url)

if model is not None:
    model.eval()
else:
    print("Failed to load the model. Please check the MEGA URL and your internet connection.")
