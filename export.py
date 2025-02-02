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


import tensorflow as tf
# 加载 Keras 模型
model = tf.keras.models.load_model('./weight/model.keras')
# 将模型保存为 TensorFlow SavedModel 格式
saved_model_path = './weight/tf_model'  # 不加扩展名
model.export(saved_model_path)  # 使用 model.export() 保存为 SavedModel 格式
print(f"模型已保存为 TensorFlow SavedModel 格式：{saved_model_path}")
