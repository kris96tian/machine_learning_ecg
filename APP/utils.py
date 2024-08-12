import numpy as np
#import wfdb
import h5py
import pandas as pd

def load_csv(file_path):
    ecg_data = pd.read_csv(file_path).values
    return ecg_data.T  # -> to (leads, time_points)

#def load_wfdb(record_name):
#    record = wfdb.rdrecord(record_name)
#    return record.p_signal.T  # -> to (leads, time_points)

def load_hdf5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        ecg_data = f[dataset_name][:]
    return ecg_data.T  # -> to (leads, time_points)

def preprocess_ecg_data(ecg_data, target_length=5300):
    #data must have 12 leads and a spec. number of time points
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
