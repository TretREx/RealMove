# import tensorflow as tf
# from keras.api.models import load_model

# # 加载 Keras 模型
# model = load_model(r"./weight/model.keras", compile=False)

# # 创建 TFLite 转换器
# converter = tf.lite.TFLiteConverter.from_keras_model(model)

# # 尝试启用资源变量标志
# converter.experimental_enable_resource_variables = True

# # 尝试禁用 TensorList 操作
# converter._experimental_lower_tensor_list_ops = False

# # 使用 Select TF Ops
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# # 进行模型转换
# tflite_model = converter.convert()

# # 保存 TFLite 模型
# with open("model.tflite", "wb") as f:
#     f.write(tflite_model)

# print("TFLite 模型已保存为 model.tflite")


# import tensorflow as tf
# # 加载 Keras 模型
# model = tf.keras.models.load_model('./weight/model.keras')
# # 将模型保存为 TensorFlow SavedModel 格式
# saved_model_path = './weight/tf_model'  # 不加扩展名
# model.export(saved_model_path)  # 使用 model.export() 保存为 SavedModel 格式
# print(f"模型已保存为 TensorFlow SavedModel 格式：{saved_model_path}")


import tensorflow as tf
import tf2onnx

# 加载 Keras 模型
keras_model = tf.keras.models.load_model('./weight/model.keras')

# 如果模型没有 output_names 属性，则手动添加
if not hasattr(keras_model, 'output_names'):
    keras_model.output_names = [f'output_{i}' for i in range(len(keras_model.outputs))]
    print("Manually added output_names:", keras_model.output_names)

# 获取模型的输入形状
input_shape = keras_model.input_shape
# 如果 batch_size 为 None，则将其固定为 1
input_shape_fixed = [1 if dim is None else dim for dim in input_shape]
print(f"Using input shape: {input_shape_fixed}")

# 定义输入签名
spec = (tf.TensorSpec(input_shape_fixed, tf.float32, name="input"),)

# 指定输出路径及 ONNX opset 版本
output_path = "./weight/model.onnx"
opset_version = 13  # 根据需要选择合适的 opset 版本

# 转换模型为 ONNX 格式
model_proto, _ = tf2onnx.convert.from_keras(
    keras_model,
    input_signature=spec,
    opset=opset_version,
    output_path=output_path
)

print(f"ONNX model has been saved to {output_path}")
