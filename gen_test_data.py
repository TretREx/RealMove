import numpy as np
import pandas as pd
import random
import os
import uuid

# 初始化函数：生成基础噪声
def generate_base_noise(time_steps):
    acc_x = np.random.normal(0, 0.01, time_steps).astype(np.float32)  # X 轴加速度（噪声）
    acc_y = np.random.normal(0, 0.01, time_steps).astype(np.float32)  # Y 轴加速度（噪声）
    acc_z = np.random.normal(9.8, 0.01, time_steps).astype(np.float32)  # Z 轴加速度（重力 + 噪声）
    gyro_x = np.random.normal(0, 0.01, time_steps).astype(np.float32)  # X 轴角速度（噪声）
    gyro_y = np.random.normal(0, 0.01, time_steps).astype(np.float32)  # Y 轴角速度（噪声）
    gyro_z = np.random.normal(0, 0.01, time_steps).astype(np.float32)  # Z 轴角速度（噪声）
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z


def generate_jumping_data(time_steps, num_samples, save_path="datasets/train/jump/"):
    """
    生成模拟跳跃数据的 CSV 文件
    """
    for sample_idx in range(num_samples):
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = generate_base_noise(time_steps)

        # 设定跳跃的关键时刻
        jump_start = random.randint(int(time_steps * 0.2), int(time_steps * 0.4))  # 起跳点
        jump_peak = jump_start + random.randint(5, 15)  # 跳跃最高点
        jump_end = jump_peak + random.randint(5, 15)  # 落地时间

        jump_strength = random.randint(1, 15) 

        for i in range(jump_start, jump_end):
            if i < jump_peak:  
                # 起跳阶段（加速上升，acc_z > 9.8）
                acc_z[i] += jump_strength * np.sin((i - jump_start) / (jump_peak - jump_start) * np.pi)
            elif i == jump_peak:
                # 最高点（短暂失重，acc_z ≈ 0）
                acc_z[i] = np.random.normal(0, 0.2)
            else:
                # 下落阶段（加速下降，acc_z < 9.8）
                acc_z[i] -= jump_strength * np.sin((i - jump_peak) / (jump_end - jump_peak) * np.pi)

            # 陀螺仪模拟（轻微姿态调整）
            gyro_x[i] += np.random.normal(0, 0.02)
            gyro_y[i] += np.random.normal(0, 0.02)

        # 数据保留三位小数
        acc_x, acc_y, acc_z = np.round(acc_x, 3), np.round(acc_y, 3), np.round(acc_z, 3)
        gyro_x, gyro_y, gyro_z = np.round(gyro_x, 3), np.round(gyro_y, 3), np.round(gyro_z, 3)

        # 存储数据到 CSV
        df = pd.DataFrame({
            "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z,
            "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z
        })
        file_name = os.path.join(save_path, f"jump_data_{sample_idx}.csv")
        df.to_csv(file_name, index=False)
        print(f"跳跃数据 {sample_idx} 已保存: {file_name}")

def generate_squat_data(time_steps, num_samples, save_path="datasets/train/squat/"):
    """
    生成模拟下蹲数据并保存为 CSV 文件。
    """
    os.makedirs(save_path, exist_ok=True)

    for sample_idx in range(num_samples):
        # 初始化数据
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = generate_base_noise(time_steps)

        # 设定下蹲关键时刻
        squat_start = random.randint(int(time_steps * 0.2), int(time_steps * 0.3))  # 开始下蹲
        squat_bottom = squat_start + random.randint(10, 20)  # 到达最低点
        squat_end = min(squat_bottom + random.randint(10, 20), time_steps - 1)  # 站起结束，确保不超出索引范围

        for i in range(squat_start, squat_end):
            if i < squat_bottom:
                # 下降阶段（acc_z < 9.8）
                acc_z[i] -= 1.5 * np.sin((i - squat_start) / (squat_bottom - squat_start) * np.pi)
            elif i == squat_bottom:
                # 最低点（acc_z 最小）
                acc_z[i] = np.random.normal(8.5, 0.2)
            else:
                # 站起阶段（acc_z > 9.8）
                acc_z[i] += 1.5 * np.sin((i - squat_bottom) / (squat_end - squat_bottom) * np.pi)

        # 数据保留三位小数
        acc_x = np.round(acc_x, 3).astype(np.float32)
        acc_y = np.round(acc_y, 3).astype(np.float32)
        acc_z = np.round(acc_z, 3).astype(np.float32)
        gyro_x = np.round(gyro_x, 3).astype(np.float32)
        gyro_y = np.round(gyro_y, 3).astype(np.float32)
        gyro_z = np.round(gyro_z, 3).astype(np.float32)

        # 保存数据
        df = pd.DataFrame({
            "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z,
            "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z
        })
        file_name = os.path.join(save_path, f"squat_data_{sample_idx}.csv")
        df.to_csv(file_name, index=False)
        print(f"下蹲数据 {sample_idx} 已保存: {file_name}")


def generate_walking_data(time_steps, num_samples, save_path="datasets/train/move/"):
    """
    生成模拟行走数据的 CSV 文件
    
    参数：
    - time_steps: 每个样本的时间步长（数据点数量）。
    - num_samples: 生成的样本数量。
    - save_path: 存储 CSV 文件的路径（文件夹）。
    """
    os.makedirs(save_path, exist_ok=True)  # 确保目录存在

    for sample_idx in range(1, num_samples + 1):
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = generate_base_noise(time_steps)

        # 模拟行走的周期性波动
        step_frequency = 2 * np.pi / time_steps  # 步频
        for i in range(time_steps):
            acc_x[i] += 0.5 * np.sin(step_frequency * i)  # 水平方向周期性变化
            acc_y[i] += 0.3 * np.cos(step_frequency * i)
            acc_z[i] += 0.2 * np.sin(step_frequency * i)  # 竖直方向微小起伏
            gyro_x[i] += 0.1 * np.sin(step_frequency * i)  # 角速度变化
            gyro_y[i] += 0.1 * np.cos(step_frequency * i)

        # 归一化并保留三位小数
        acc_x, acc_y, acc_z = np.round(acc_x, 3), np.round(acc_y, 3), np.round(acc_z, 3)
        gyro_x, gyro_y, gyro_z = np.round(gyro_x, 3), np.round(gyro_y, 3), np.round(gyro_z, 3)

        # 存储数据到 CSV
        df = pd.DataFrame({
            "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z,
            "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z
        })
        file_name = os.path.join(save_path, f"move_data_{sample_idx}.csv")
        df.to_csv(file_name, index=False)
        print(f"行走数据 {sample_idx} 已生成并保存: {file_name}")



if __name__=="__main__":
    num_samples = 100  # 生成100组数据
    time_steps = 50  # 时间步长（与模型输入一致）
    generate_jumping_data(time_steps, num_samples)
    generate_squat_data(time_steps, num_samples)
    generate_walking_data(time_steps, num_samples)
