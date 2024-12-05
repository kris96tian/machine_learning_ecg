# ECG Heart Attack Prediction App

### Overview
The **ECG Heart Attack Prediction App** is a machine-learning-powered tool designed to assess heart attack risk based on ECG data. Using a pre-trained deep learning model, the application analyzes ECG samples provided in either CSV or HDF5 format and returns a prediction score along with a risk level (Low, Medium, High).

### Features
- **Deep Learning Model**: Leverages a custom convolutional neural network (CNN) to process 12-lead ECG data.
- **Data Input Formats**: Supports ECG data in CSV and HDF5 file formats.
- **Real-Time Risk Assessment**: Provides a risk percentage and categorization (Low, Medium, High).
- **User-Friendly Interface**: Powered by [Gradio](https://gradio.app/) for easy interaction.

### How It Works
1. **Upload ECG Data**: Provide your ECG file in CSV or HDF5 format. You can also use the file 'exampledata.csv' to try.
2. **Preprocessing**: The app normalizes and formats your ECG data to ensure compatibility with the model.
3. **Prediction**: The model analyzes the data and outputs:
   - A percentage risk score.
   - A risk level based on the prediction:
     - **Low Risk**: ≤ 25%.
     - **Medium Risk**: 26%–75%.
     - **High Risk**: > 75%.

### Usage
1.1. Clone the repository and ensure dependencies are installed.
1.2. Launch the app with the following command:
   ```bash
   python app.py
   ```
2. Use the deployed webapp at https://huggingface.co/spaces/kral2796/heart-attack-prediction/blob/main/app.py

### Requirements
- **Python 3.7+**
- Python libraries: `torch`, `numpy`, `pandas`, `h5py`, `gradio`

### File Formats
- **CSV**: A CSV file containing ECG data, with rows representing leads and columns representing data points.
- **HDF5**: A dataset containing ECG data; specify the dataset name when uploading.

### Model Architecture
The deep learning model (`ECGCNN`) consists of:
- 3 convolutional layers for feature extraction.
- A fully connected layer for classification.
- Activation functions: ReLU and Sigmoid for final risk probability.

### Example Output
- **Prediction**: `85.42%`
- **Risk Level**: `High Risk`

### Credits
Developed by **Kristian Alikaj**  
- [GitHub Repository](https://github.com/kris96tian/machine_learning_ecg)

### Disclaimer
This app is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
