import os
import json
import numpy as np
import pandas as pd
import joblib
import logging
import random
import tensorflow as tf

from keras.api.models import Sequential, load_model
from keras.api.layers import LSTM, Dense, Dropout, Masking
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# -------------------------- 日志与随机种子设置 --------------------------
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# -------------------------- 读取配置文件 --------------------------
config_path = './weight/config.json'
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    logging.info(f"Loaded configuration from {config_path}")
except Exception as e:
    logging.error(f"Error reading config file: {e}")
    raise e

data_dir = config["data_dir"]               # 原始训练数据存放目录
activity_dirs = config["activity_dirs"]     # 各动作

# -------------------------- 数据加载与预处理 --------------------------
def load_data():
    """
    从各 activity 文件夹中读取 CSV 文件，返回样本列表和标签列表。
    同时构建 label -> activity 的映射字典。
    """
    X, y = [], []
    activity_map = {}
    label = 0
    for activity, folder in activity_dirs.items():
        activity_map[label] = activity.capitalize() # 首字母大写
        activity_dir = os.path.join(data_dir, folder)
        if not os.path.exists(activity_dir):
            logging.warning(f"Directory {activity_dir} does not exist. Skipping.")
            continue

        for file in os.listdir(activity_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(activity_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    if df.empty:
                        logging.warning(f"File {file_path} is empty. Skipping.")
                        continue
                    X.append(df.values)
                    y.append(label)
                except Exception as ex:
                    logging.error(f"Error reading {file_path}: {ex}")
        label += 1

    X, y = np.array(X), np.array(y)
    logging.info(f"Loaded {len(X)} samples from {len(activity_map)} activities.")
    return X, y, activity_map

def pad_sequences(X, max_len=None):
    """
    将所有序列（样本）补零到相同长度。
    :param X: 列表或数组，每个元素形状为 (time_steps, num_features)
    :param max_len: 指定的最大长度，如果为 None，则自动取所有样本中的最大长度
    :return: 形状为 (num_samples, max_len, num_features) 的 ndarray
    """
    if max_len is None:
        max_len = max(sample.shape[0] for sample in X)# 取样本中最长的长度
    num_features = X[0].shape[1]# 取样本的特征数
    X_padded = np.zeros((len(X), max_len, num_features))# 初始化补零后的样本
    for i, sample in enumerate(X):# 遍历每个样本
        length = sample.shape[0]# 取样本的长度
        if length > max_len:# 如果长度大于最大长度
            X_padded[i] = sample[:max_len, :]# 直接截取
            X_padded[i] = sample[:max_len, :]
        else:
            X_padded[i, :length, :] = sample
    return X_padded# 返回补零后的样本

def preprocess_data(X, scaler=None, fit_scaler=True):
    """
    对数据进行归一化处理。
    :param X: ndarray, 形状 (num_samples, time_steps, num_features)
    :param scaler: 如果已有归一化器，则直接使用；否则将新建一个 StandardScaler
    :param fit_scaler: 是否在数据上拟合归一化器（仅对训练数据拟合）
    :return: 归一化后的数据以及 scaler
    """
    num_samples, time_steps, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)# 展开为 (num_samples * time_steps, num_features)
    
    if scaler is None:
        scaler = StandardScaler()  # 如有需要可切换为 MinMaxScaler() 等
    
    if fit_scaler:
        scaler.fit(X_reshaped)
        logging.info("Fitted scaler on training data.")
    
    X_scaled = scaler.transform(X_reshaped)# 归一化
    X_scaled = X_scaled.reshape(num_samples, time_steps, num_features)# 还原为 (num_samples, time_steps, num_features)
    
    return X_scaled, scaler

# -------------------------- 模型构建 --------------------------
def create_model(input_shape, num_classes):
    """
    构建 LSTM 模型。
    :param input_shape: 输入数据形状（time_steps, num_features）
    :param num_classes: 分类数
    :return: 编译后的模型
    """
    model = Sequential([
        # Masking 层用于忽略补零部分
        Masking(mask_value=0.0, input_shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    logging.info("Created LSTM model.")
    return model

# -------------------------- 训练与评估 --------------------------
def train_and_evaluate(epochs=50, batch_size=32):
    # 加载数据
    X_raw, y, activity_map = load_data()
    
    # 统一序列长度
    X_padded = pad_sequences(X_raw)
    logging.info(f"Padded sequences shape: {X_padded.shape}")
    
    # 划分训练集和测试集（分层抽样保证各类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.2, random_state=seed, stratify=y)
    logging.info(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    
    # 在训练数据上拟合 scaler，再对训练集与测试集进行转换
    X_train, scaler = preprocess_data(X_train, scaler=None, fit_scaler=True)
    X_test, _ = preprocess_data(X_test, scaler=scaler, fit_scaler=False)
    
    # 构建模型（50，6）
    model = create_model(input_shape=X_train.shape[1:], num_classes=len(activity_map))
    
    # 计算类别权重，防止类别不平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    logging.info(f"Class weights: {class_weight_dict}")
    
    # 定义回调函数：EarlyStopping、ModelCheckpoint 和 ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, 
                                   restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint('./weight/best_model.keras', monitor='val_loss', 
                                 save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                                  min_lr=1e-6, verbose=1)
    
    # 模型训练
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, checkpoint, reduce_lr],
                        class_weight=class_weight_dict)
    
    # 模型评估
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # 进一步评估：输出分类报告与混淆矩阵
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    report = classification_report(y_test, y_pred, 
                                   target_names=[activity_map[i] for i in sorted(activity_map.keys())])
    conf_matrix = confusion_matrix(y_test, y_pred)
    logging.info("Classification Report:\n" + report)
    logging.info("Confusion Matrix:\n" + str(conf_matrix))
    
    return model, scaler, activity_map, history

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    model, scaler, activity_map, history = train_and_evaluate(epochs=100, batch_size=32)
    # 保存最终模型和归一化器
    model.save('./weight/model.keras')
    joblib.dump(scaler, './weight/scaler.pkl')
    logging.info("Saved final model and scaler.")
