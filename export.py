import tensorflow as tf

# 加载 Keras 模型（H5 格式）
h5_model_path = "./weight/model.h5"  # 替换成你的 H5 文件路径
model = tf.keras.models.load_model(h5_model_path)
print(f"成功加载模型：{h5_model_path}")

# 创建 TFLite 转换器
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 启用资源变量支持
converter.experimental_enable_resource_variables = True

# 使用 SELECT_TF_OPS，允许使用所有 TensorFlow 操作
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

# 转换模型
tflite_model = converter.convert()
print("模型转换成功！")

# 保存 TFLite 模型
tflite_model_path = "./weight/model.tflite"  # 输出文件路径
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print(f"TFLite 模型已保存至：{tflite_model_path}")

# 验证转换后的 TFLite 模型
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("输入张量信息:", input_details)
print("输出张量信息:", output_details)
