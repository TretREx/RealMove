import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

class ActivityPredictor:
    def __init__(self, model_type='keras', config_path='./weight/config.json'):
        """
        初始化 ActivityPredictor 类
        :param model_type: 'keras' 或 'tflite'，选择使用 Keras 或 TFLite 模型
        :param config_path: JSON 配置文件路径
        """
        self.model_type = model_type

        # 读取配置文件
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.model_paths = config["model_paths"]
        self.scaler_path = config["scaler_path"]
        self.activity_map = {int(k): v for k, v in config["activity_map"].items()}  # 转换 key 为 int

        # 加载标准化器
        print("🔄 加载标准化器...")
        self.scaler = joblib.load(self.scaler_path)
        print("✅ Scaler 加载成功！")

        # 加载模型
        self.model = None
        self.tflite_model = None
        self.load_model()

    def load_model(self):
        """ 加载 Keras 或 TFLite 模型 """
        try:
            if self.model_type == 'keras':
                print("🔄 加载 Keras 模型...")
                self.model = tf.keras.models.load_model(self.model_paths['keras'])
                print("✅ Keras 模型加载成功！")
            elif self.model_type == 'tflite':
                print("🔄 加载 TFLite 模型...")
                with open(self.model_paths['tflite'], 'rb') as f:
                    self.tflite_model = f.read()
                print("✅ TFLite 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

    def preprocess_input(self, file_path):
        """ 读取 CSV 文件并进行标准化 """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ 文件未找到: {file_path}")

        df = pd.read_csv(file_path)
        print(f"📂 加载数据: {file_path} (Shape: {df.shape})")

        # 确保数据格式正确
        if df.shape != (50, 6):
            raise ValueError(f"❌ 输入数据格式错误，期望形状: (50, 6)，实际形状: {df.shape}")

        # 进行标准化
        scaled_data = self.scaler.transform(df.values)
        return np.expand_dims(scaled_data, axis=0)  # 变成 (1, 50, 6) 以适配模型输入

    def predict(self, file_path):
        """ 预测活动 """
        input_data = self.preprocess_input(file_path)

        if self.model_type == 'keras':
            return self._predict_keras(input_data)
        elif self.model_type == 'tflite':
            return self._predict_tflite(input_data)

    def _predict_keras(self, input_data):
        """ 使用 Keras 模型进行预测 """
        if self.model is None:
            raise ValueError("❌ Keras 模型未加载")

        print("🔄 进行 Keras 预测...")
        prediction = self.model.predict(input_data)
        return self._postprocess_prediction(prediction)

    def _predict_tflite(self, input_data):
        """ 使用 TFLite 模型进行预测 """
        if self.tflite_model is None:
            raise ValueError("❌ TFLite 模型未加载")

        print("🔄 进行 TFLite 预测...")
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])
        return self._postprocess_prediction(prediction)

    def _postprocess_prediction(self, prediction):
        """ 解析模型输出 """
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]

        activity = self.activity_map.get(predicted_class, "Unknown")
        print(f"🎯 预测结果: {activity} (Confidence: {confidence:.4f})")
        return activity, confidence

# 运行测试
if __name__ == "__main__":
    input_file_path = 'datasets/train/move/move_data_1.csv'  # 替换为实际文件路径

    # 使用 Keras 模型进行预测
    predictor_keras = ActivityPredictor(model_type='keras')
    predictor_keras.predict(input_file_path)

    # 使用 TFLite 模型进行预测
    predictor_tflite = ActivityPredictor(model_type='tflite')
    predictor_tflite.predict(input_file_path)