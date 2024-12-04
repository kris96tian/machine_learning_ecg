import torch
import torch.nn as nn

class ECGCNN(nn.Module):
    def __init__(self):
        super(ECGCNN, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 662, 128)  # 128 * 662 = 84736, adjust based on your input data shape
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through convolution and pooling layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path='model.ph'):
    try:
        # Attempt to load the checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Initialize the model
        model = ECGCNN()
        
        # Check if state_dict exists in checkpoint or directly load
        checkpoint_state_dict = checkpoint.get('state_dict', checkpoint)
        model_state_dict = model.state_dict()
        
        # Filter matching keys to load into the model
        filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        
        # Load the filtered state dict into the model
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()  # Switch model to evaluation mode
        return model
    except FileNotFoundError:
        print(f"Error: The model file '{model_path}' was not found.")
    except Exception as e:
        print(f"Error loading model: {e}")
    return None

# Load the model
model = load_model()

# Check if the model loaded successfully
if model is not None:
    print("Model loaded successfully!")
else:
    print("Failed to load the model")
