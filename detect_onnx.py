import os
import json
import random
import time  # 用于计时
import numpy as np
import pandas as pd
import joblib
import onnxruntime

# -------------------- 1. 加载配置文件 --------------------
config_path = './weight/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

data_dir = config["data_dir"]
activity_dirs = config["activity_dirs"]
model_paths = config["model_paths"]
scaler_path = config["scaler_path"]
activity_map = config["activity_map"]

# 假设 ONNX 模型保存在与 keras 模型同目录下，文件名为 model.onnx
onnx_model_path = os.path.join(os.path.dirname(model_paths["keras"]), "model.onnx")
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

# -------------------- 2. 加载归一化器 --------------------
scaler = joblib.load(scaler_path)
print("Loaded scaler from:", scaler_path)

# -------------------- 3. 加载 ONNX 模型 --------------------
session = onnxruntime.InferenceSession(onnx_model_path)
print("Loaded ONNX model from:", onnx_model_path)

# -------------------- 4. 从数据目录中随机选取一个 CSV 文件 --------------------
# 随机选择一个活动
activity = random.choice(list(activity_dirs.keys()))
activity_folder = os.path.join(data_dir, activity_dirs[activity])
csv_files = [os.path.join(activity_folder, file) for file in os.listdir(activity_folder)
             if file.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {activity_folder}")

sample_file = random.choice(csv_files)
print("Selected sample file:", sample_file)

# 读取 CSV 文件
def load_sample(csv_file):
    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"File {csv_file} is empty.")
    return df.values

sample = load_sample(sample_file)
print("Original sample shape:", sample.shape)

# -------------------- 5. 数据预处理 --------------------
# 假设模型训练时要求固定的时间步长（例如 50）和固定的特征数（例如 6）。
# 如时间步长不足，则补零；过长则截断。
def pad_sample(sample, target_length=50):
    timesteps, features = sample.shape
    if timesteps >= target_length:
        return sample[:target_length, :]
    else:
        padded = np.zeros((target_length, features))
        padded[:timesteps, :] = sample
        return padded

# 设置固定长度，例如 50
fixed_length = 50
sample_padded = pad_sample(sample, target_length=fixed_length)
print("Padded sample shape:", sample_padded.shape)

# 使用加载好的 scaler 进行归一化（Scaler 训练时是对二维数据拟合的）
def preprocess_sample(sample, scaler):
    # sample shape: (time_steps, features)
    sample_scaled = scaler.transform(sample)
    # 增加 batch 维度：最终形状 (1, time_steps, features)
    return np.expand_dims(sample_scaled, axis=0).astype(np.float32)

input_data = preprocess_sample(sample_padded, scaler)
print("Input data shape for ONNX inference:", input_data.shape)

# -------------------- 6. ONNX 推理 --------------------
# 获取 ONNX 模型的输入名称
input_name = session.get_inputs()[0].name
print("ONNX model input name:", input_name)

# 执行推理并计时
start_time = time.time()  # 开始计时
outputs = session.run(None, {input_name: input_data})
end_time = time.time()    # 推理结束计时

# 计算推理耗时
inference_time = end_time - start_time
print("Inference time: {:.4f} seconds".format(inference_time))

# 假设模型输出为 softmax 概率分布
predictions = outputs[0]
print("Raw predictions:", predictions)

# 获取预测的类别索引
predicted_index = int(np.argmax(predictions, axis=1)[0])
# 注意：config 中 activity_map 的 key 是字符串，所以转换索引为字符串进行映射
predicted_activity = activity_map.get(str(predicted_index), "Unknown")
print("Predicted activity:", predicted_activity)
