import onnxruntime as ort
import numpy as np

# 创建一个简单的 ONNX 模型路径（替换为你的模型路径）
model_path = "./weight/model.onnx"

# 创建推理会话，并优先使用 GPU
session = ort.InferenceSession(
    model_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# 打印实际使用的执行提供者
print("Execution Providers:", session.get_providers())

# 准备输入数据（假设模型输入形状为 (1, 50, 6)）
input_name = session.get_inputs()[0].name
input_data = np.random.rand(1, 50, 6).astype(np.float32)

# 执行推理
outputs = session.run(None, {input_name: input_data})
print("Inference completed successfully!")