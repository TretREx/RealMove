import os
import json
import random
import time
import numpy as np
import pandas as pd
import joblib
import onnxruntime

def run_inference(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_dir = config["data_dir"]
    activity_dirs = config["activity_dirs"]
    model_paths = config["model_paths"]
    scaler_path = config["scaler_path"]
    activity_map = config["activity_map"]

    onnx_model_path = os.path.join(os.path.dirname(model_paths["keras"]), "model.onnx")
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

    scaler = joblib.load(scaler_path)

    session = onnxruntime.InferenceSession(onnx_model_path)

    activity = random.choice(list(activity_dirs.keys()))
    activity_folder = os.path.join(data_dir, activity_dirs[activity])
    csv_files = [os.path.join(activity_folder, file) for file in os.listdir(activity_folder)
                 if file.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {activity_folder}")

    sample_file = random.choice(csv_files)

    def load_sample(csv_file):
        df = pd.read_csv(csv_file)
        if df.empty:
            raise ValueError(f"File {csv_file} is empty.")
        return df.values

    sample = load_sample(sample_file)

    def pad_sample(sample, target_length=50):
        timesteps, features = sample.shape
        if timesteps >= target_length:
            return sample[:target_length, :]
        else:
            padded = np.zeros((target_length, features))
            padded[:timesteps, :] = sample
            return padded

    fixed_length = 50
    sample_padded = pad_sample(sample, target_length=fixed_length)


    def preprocess_sample(sample, scaler):
        sample_scaled = scaler.transform(sample)
        return np.expand_dims(sample_scaled, axis=0).astype(np.float32)

    input_data = preprocess_sample(sample_padded, scaler)


    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})


    predictions = outputs[0]
    predicted_index = int(np.argmax(predictions, axis=1)[0])


    predicted_activity = activity_map.get(str(predicted_index), "Unknown")

    return predicted_activity, predictions
