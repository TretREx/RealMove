import numpy as np
import tensorflow as tf
import pandas as pd

# 加载 TFLite 模型
tflite_model_path = "model.tflite"  # 替换为你的 TFLite 模型路径
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

# 分配张量
interpreter.allocate_tensors()

# 获取输入输出张量的详细信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# print("输入张量信息:", input_details)
# print("输出张量信息:", output_details)

# 从 CSV 文件加载数据（假设 CSV 文件每一行是一个样本，每列是一个特征）
csv_file_path = "datasets\\train\\move\\move_data_1.csv"  # 替换为你的 CSV 文件路径
data = pd.read_csv(csv_file_path)

# 将数据转换为 NumPy 数组并确保类型为 float32
input_data = data.values.astype(np.float32)

# 检查是否需要调整数据形状，例如将其调整为 (1, 50, 6)
input_data = input_data.reshape((1, 50, 6))  # 根据模型需要的形状进行调整

# 将数据填充到输入张量
interpreter.set_tensor(input_details[0]['index'], input_data)

# 执行推理
interpreter.invoke()

# 获取输出数据
output_data = interpreter.get_tensor(output_details[0]['index'])

# 假设输出是一个概率分布或分类分数，选择概率最高的类别
predicted_class = np.argmax(output_data)

print(predicted_class)
# 映射类别到动作标签
actions = {1: 'jump', 2: 'move', 3: 'squat'}
predicted_action = actions.get(predicted_class, 'unknown')

print(f"预测的动作是: {predicted_action}")
