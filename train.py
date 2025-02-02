import os
import numpy as np
import pandas as pd
import json
import joblib
from keras.api.models import Sequential, load_model
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# 读取 `config.json`
with open('./weight/config.json', 'r') as f:
    config = json.load(f)

data_dir = config["data_dir"]
activity_dirs = config["activity_dirs"]

def load_and_preprocess_data():
    X, y = [], []
    activity_map = {}

    label = 0
    for activity, folder in activity_dirs.items():
        activity_map[label] = activity.capitalize()
        activity_dir = os.path.join(data_dir, folder)

        for file in os.listdir(activity_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(activity_dir, file))
                X.append(df.values)
                y.append(label)
        label += 1

    X, y = np.array(X), np.array(y)

    # 归一化
    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, X.shape[-1]))  # 只对整个数据集拟合一次
    X_scaled = np.array([scaler.transform(sample) for sample in X])

    return X_scaled, y, scaler, activity_map

def create_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_predict(epochs=50):
    X_scaled, y, scaler, activity_map = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = create_model(X_train.shape[1:], len(activity_map))

    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # 早停
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), 
              callbacks=[early_stopping], class_weight=class_weight_dict)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"测试集上的损失值: {loss:.4f}, 准确率: {accuracy:.4f}")

    return model, scaler, activity_map

if __name__ == "__main__":
    model, scaler, activity_map = train_and_predict(epochs=100)
    model.save('./weight/model.keras')
    joblib.dump(scaler, './weight/scaler.pkl')
