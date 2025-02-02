import tensorflow as tf
import keras
# 打印 TensorFlow 版本
print("TensorFlow version:", tf.__version__)

# 打印 Keras 版本
print("Keras version:", keras.__version__)

# 检查 GPU 是否可用
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# 简单的张量运算测试
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
result = tf.add(a, b)
print("Tensor addition result:", result.numpy())