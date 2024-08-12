import numpy as np
import h5py
import pandas as pd

def load_csv(file_path):
    ecg_data = pd.read_csv(file_path).values
    return ecg_data.T  # -> to (leads, time_points)

def load_hdf5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        ecg_data = f[dataset_name][:]
    return ecg_data.T  # -> to (leads, time_points)
    
def preprocess_ecg_data(ecg_data, target_length=5300):
    num_leads, num_points = ecg_data.shape
    
    if num_leads != 12:
        raise ValueError(f"Expected 12 leads, but got {num_leads} leads.")
    
    # adjusting data to have exactly `target_length` time points
    if num_points > target_length:
        ecg_data = ecg_data[:, :target_length] 
    elif num_points < target_length:
        padding = target_length - num_points
        ecg_data = np.pad(ecg_data, ((0, 0), (0, padding)), 'constant')  
    
    return ecg_data


    
def load_data(uploaded_file, file_type, dataset_name, target_length):
    file_path = uploaded_file
        if file_type == 'csv':
        ecg_data = load_csv(file_path)
    elif file_type == 'hdf5':
        ecg_data = load_hdf5(file_path, dataset_name)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or HDF5 file.")
    
    ecg_data = preprocess_ecg_data(ecg_data, target_length)
    
    return ecg_data
