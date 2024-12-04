import torch
import torch.nn as nn

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

def load_model():
    checkpoint = torch.load('model.ph', map_location=torch.device('cpu'))
    model = ECGCNN()  

    checkpoint_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}

    model.load_state_dict(filtered_state_dict, strict=False)
    return model


# Let's go!
model = load_model()

if model is not None:
    model.eval()
else:
    print("Failed to load the model")