import onnxruntime as ort

# 创建推理会话时指定 GPU 提供者
try:
    session = ort.InferenceSession(
        "./weight/model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print("Execution Providers:", session.get_providers())
except RuntimeError as e:
    print("Error:", e)