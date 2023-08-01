# IMPORTS:
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.signal import detrend
import pandas as pd
from torch.utils.data import random_split
from sklearn.preprocessing import MinMaxScaler


# MY DATASET.CLASS:
class ECGData(Dataset):
    def __init__(self):
        rwma_labels = np.loadtxt(
            "/home/ngsci/datasets/silent-cchs-ecg/csv/rwma-outcomes.csv",
            delimiter=",",
            dtype=np.float32,
            skiprows=1,
        )
        npy_filepath = "/home/ngsci/datasets/silent-cchs-ecg/npy"
        dir_list = os.listdir(npy_filepath)
        npy_arrays = []
        for each in dir_list:
            file = f"{npy_filepath}/{each}"
            npy_arrays.append(np.load(file))
        stacked = np.stack(npy_arrays, axis=1)
        self.X = torch.from_numpy(stacked)
        self.y = torch.from_numpy(rwma_labels[:, 1:])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx, :, :, :], self.y[idx])


# DATA SPLIT / PREPROCESS / PREPARE / INITIATE...
data = ECGData() #instanz of above Custom Class

###  DATA SPLIT FUNCKTION and INIT.
def split_data(data):
    indexRange = torch.arange(0, len(data))
    train_size = int(0.8 * len(data))
    train_split, test_split = random_split(
        indexRange, [train_size, len(data) - train_size]
    )
    X_train = data[train_split][0].squeeze(dim=2)
    X_test = data[test_split][0].squeeze(dim=2)
    y_train = data[train_split][1]
    y_test = data[test_split][1]
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_data(data)


## Data NORMALIZATION
##
def preprocess_data(X_train, X_test):
    # Min-max scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
        X_test.shape
    )

    return X_train_scaled, X_test_scaled


X_train_preprocessed, X_test_preprocessed = preprocess_data(X_train, X_test)



X_train = torch.tensor(
    X_train_preprocessed[:, :, :5000], dtype=torch.float32, requires_grad=False
)
X_test = torch.tensor(
    X_test_preprocessed[:, :, :5000], dtype=torch.float32, requires_grad=False
)
y_train = torch.tensor(y_train[:5000], requires_grad=False)
y_test = torch.tensor(y_test[:5000], requires_grad=False)


train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for inputs, targets in train_loader:
    print(inputs, targets)
    break


# CNN MODEL CLASS & Instance INIT.:
import torch.optim as optim

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 2499, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Create an instance of the model
cnn_model = CNNModel()

from sklearn.metrics import confusion_matrix

def accuracy(outputs, targets):
    predicted = (outputs > 0.5).float()
    correct = (predicted == targets).float()
    acc = correct.sum() / len(targets)
    return acc.item()

# Modify the target labels to binary values using a threshold
threshold = 0.5
y_train_binary = (y_train >= threshold).float()
y_test_binary = (y_test >= threshold).float()

# Define the binary cross-entropy loss
criterion = nn.BCELoss()

# Define the optimizer
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    # Training
    cnn_model.train()
    train_loss = 0
    train_correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()
        
        # Compute training accuracy
        predicted = (outputs >= threshold).float()
        train_correct += (predicted.squeeze() == targets.squeeze()).sum().item()
        total += targets.size(0)
        
        train_loss += loss.item()

    train_acc = train_correct / total
    train_loss_history.append(train_loss / len(train_loader))
    train_acc_history.append(train_acc)
    
    # Validation
    cnn_model.eval()
    val_loss = 0
    val_correct = 0
    total = 0
    
    for inputs, targets in test_loader:
        outputs = cnn_model(inputs)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        
        # Compute validation accuracy
        predicted = (outputs >= threshold).float()
        val_correct += (predicted.squeeze() == targets.squeeze()).sum().item()
        total += targets.size(0)
        
        val_loss += loss.item()

    val_acc = val_correct / total
    val_loss_history.append(val_loss / len(test_loader))
    val_acc_history.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_history[-1]:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss_history[-1]:.4f}, Val Acc: {val_acc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test_binary, (cnn_model(X_test) >= threshold).float()))

