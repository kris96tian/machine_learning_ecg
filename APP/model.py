import torch
import torch.nn as nn
import requests
from io import BytesIO

class ECGCNN(nn.Module):
    def __init__(self):
        super(ECGCNN, self).__init__()
        # Update these to match the saved model architecture
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 662, 1024)  
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def load_model_from_dropbox(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  
        buffer = BytesIO(response.content)
        model = ECGCNN()
        model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
        
        model.eval()  
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Example Dropbox link
dropbox_url = 'https://www.dropbox.com/scl/fi/f0nehqdr5i2salflap0gg/model.pth?rlkey=p38eox3ci5w8ky1abhc0uz2cx&st=nmcqw6cb&dl=1'

model = load_model_from_dropbox(dropbox_url)
if model is None:
    raise RuntimeError("Failed to load the model.")
