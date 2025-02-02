import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf

# 数据预处理函数
def preprocess_input(file_path, scaler):
    # 读取数据文件
    df = pd.read_csv(file_path)
    
    # 对数据进行标准化
    scaled_data = scaler.transform(df.values)  # 使用训练时的标准化器
    
    # 将数据转换为模型输入格式
    return np.expand_dims(scaled_data, axis=0)

# 预测函数
def predict_activity(file_path, model, scaler):
    # 对输入数据进行预处理
    input_data = preprocess_input(file_path, scaler)
    
    # 使用模型进行预测
    prediction = model.predict(input_data)
    
    # 获取预测的类别
    predicted_class = np.argmax(prediction, axis=1)
    
    # 获取对应的置信度（最大概率）
    confidence = np.max(prediction, axis=1)
    
    # 映射到活动名称
    activity_map = {0: 'Jumping', 1: 'Squatting', 2: 'Moving'}
    
    return activity_map[predicted_class[0]], confidence[0]

# 主程序
if __name__ == "__main__":
    # 加载模型时使用 .keras 格式
    model = tf.keras.models.load_model('./weight/model.keras')
    # model = load_model('model.h5')  # 加载Keras模型
    scaler = joblib.load('./weight/scaler.pkl')  # 加载标准化器
    # 假设我们有一个新的文件需要预测
    input_folder_path = 'datasets\\train\\move'  # 你可以替换成实际路径
    for filename in os.listdir(input_folder_path):
        input_file_path = os.path.join(input_folder_path, filename)
        
        # 确保是文件而不是子文件夹
        if os.path.isfile(input_file_path):
            print(f"正在处理文件: {input_file_path}")
            
            # 预测当前文件
            predicted_activity, confidence = predict_activity(input_file_path, model, scaler)
            print(f"预测结果 for {filename}: {predicted_activity} (Confidence: {confidence:.4f})")
