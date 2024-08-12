import torch
import torch.nn as nn
from mega import Mega
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

# load model from MEGA
def load_model_from_mega(url):
    try:     
        mega = Mega()
        m = mega.login()
        file = m.download_url(url)
        
        model = ECGCNN()
        checkpoint = torch.load(file, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)

        
        return model
    except Exception as e:
      
        return None


# Load the model
mega_url = 'https://mega.nz/file/eMQyUbTL#rN13DEjWCpqp0RUYoA2fVyl4xeaErlbkj8z3WGI7gUg'
model = load_model_from_mega(mega_url)
model.eval()
    
