from keras.api.models import load_model

# 加载 Keras 模型
net = load_model(r"./weight/model.keras", compile=False)

# 获取输入和输出信息
print("Inputs info:", net.inputs)
print("Inputs info:", net.outputs)
print("Inputs names:", [input.name for input in net.inputs])
print("Output names:", [output.name for output in net.outputs])
