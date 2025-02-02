import os
import numpy as np
import pandas as pd
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from keras.api.models import load_model
import keras

# 设置文件路径
data_dir = './datasets/train/'
activity_dirs = {
    'jump': 'jump',
    'squat': 'squat',
    'move': 'move'
}

# 加载数据的函数
def load_and_preprocess_data():
    X = []
    y = []
    activity_map = {}  # To store activity label names dynamically

    # 动态加载每个活动的数据
    label = 0
    for activity, folder in activity_dirs.items():
        activity_map[label] = activity.capitalize()  # Store activity name in the map
        activity_dir = os.path.join(data_dir, folder)
        
        for file in os.listdir(activity_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(activity_dir, file))
                X.append(df.values)
                y.append(label)  # Assign the current label for this activity
        label += 1  # Increment the label for the next activity
    
    # 转换为 numpy 数组
    X = np.array(X)
    y = np.array(y)

    # 特征缩放：使用训练集的 `StandardScaler`
    scaler = StandardScaler()
    X_scaled = []
    for i in range(X.shape[0]):
        X_scaled.append(scaler.fit_transform(X[i]))  # 对每个样本进行归一化
    X_scaled = np.array(X_scaled)

    return X_scaled, y, scaler, activity_map

# 定义 LSTM 模型
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))  # 第一层 LSTM
    model.add(Dropout(0.2))
    model.add(LSTM(32))  # 第二层 LSTM
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))  # 输出层，动态类别数量

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 训练模型并返回模型和标准化器
def train_and_predict(epochs=50):
    # 训练数据加载与预处理
    X_scaled, y, scaler, activity_map = load_and_preprocess_data()

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 创建和训练模型
    model = create_model(X_train.shape[1:], len(activity_map))  # Use dynamic class count
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    
    # 模型评估
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"测试集上的损失值: {loss:.4f}, 准确率: {accuracy:.4f}")

    # 保存训练的标准化器
    return model, scaler, activity_map

# 加载并标准化单个数据文件
def preprocess_input(file_path, scaler):
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 特征缩放：将每个传感器的值归一化
    scaled_data = scaler.transform(df.values)  # 使用训练集的 scaler 进行标准化

    # 添加批次维度并返回
    return np.expand_dims(scaled_data, axis=0)

# 预测代码
def predict_activity(model, scaler, input_data, activity_map):
    # 使用训练时的 scaler 对数据进行标准化
    input_data = preprocess_input(input_data, scaler)
    
    # 进行预测
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)  # 得到最大概率的类
    predicted_activity = activity_map[predicted_class[0]]  # 映射到活动名称
    return predicted_activity

if __name__ == "__main__":
    # 训练并保存模型和标准化器
    model, scaler, activity_map = train_and_predict()
    # 保存模型
    model.save('./weight/model.keras')
    # keras.saving.save_model(model, './weight/model.h5')
    # 保存标准化器
    joblib.dump(scaler, './weight/scaler.pkl')  # 保存Scaler
    
    # # 假设我们有一个新的文件需要预测
    # input_folder_path = 'datasets\\train\\move'  # 你可以替换成实际路径
    # for filename in os.listdir(input_folder_path):
    #     input_file_path = os.path.join(input_folder_path, filename)
        
    #     # 确保是文件而不是子文件夹
    #     if os.path.isfile(input_file_path):
    #         print(f"正在处理文件: {input_file_path}")
            
    #         # 预测当前文件
    #         predicted_activity = predict_activity(model, scaler, input_file_path, activity_map)
    #         print(f"预测结果 for {filename}: {predicted_activity}")
