from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import os
from model import model
from utils import load_csv, load_hdf5, preprocess_ecg_data

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_type = request.form.get('file_type')
    dataset_name = request.form.get('dataset_name')
    target_length = int(request.form.get('target_length', 5300))

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        temp_file_path = os.path.join('/tmp', file.filename)
        file.save(temp_file_path)

        if file_type == 'csv':
            ecg_data = load_csv(temp_file_path)
        elif file_type == 'hdf5':
            ecg_data = load_hdf5(temp_file_path, dataset_name)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        ecg_data = preprocess_ecg_data(ecg_data, target_length)

        ecg_data_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(ecg_data_tensor)
            prediction = torch.sigmoid(output).item()
            prediction_percentage = prediction * 100

        return jsonify({'prediction': f'{prediction_percentage:.2f}%'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == '__main__':
    app.run(debug=True)
