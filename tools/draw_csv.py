import pandas as pd
import matplotlib.pyplot as plt

def read_csv_file(file_path, save_path=None):
    data = pd.read_csv(file_path)

    # 计算所有列的全局最小值和最大值
    global_min = data.min().min()
    global_max = data.max().max() * 3 / 2

    # 创建一个子图的布局
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # 绘制每个传感器的时间序列数据，并统一 Y 轴量程
    axes[0, 0].plot(data['acc_x'], color='b')
    axes[0, 0].set_title('Acc X')
    axes[0, 0].set_ylim(global_min, global_max)

    axes[0, 1].plot(data['acc_y'], color='g')
    axes[0, 1].set_title('Acc Y')
    axes[0, 1].set_ylim(global_min, global_max)

    axes[1, 0].plot(data['acc_z'], color='r')
    axes[1, 0].set_title('Acc Z')
    axes[1, 0].set_ylim(global_min, global_max)

    axes[1, 1].plot(data['gyro_x'], color='c')
    axes[1, 1].set_title('Gyro X')
    axes[1, 1].set_ylim(global_min, global_max)

    axes[2, 0].plot(data['gyro_y'], color='m')
    axes[2, 0].set_title('Gyro Y')
    axes[2, 0].set_ylim(global_min, global_max)

    axes[2, 1].plot(data['gyro_z'], color='y')
    axes[2, 1].set_title('Gyro Z')
    axes[2, 1].set_ylim(global_min, global_max)

    # 自动调整子图布局
    plt.tight_layout()

    # 显示图表
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存至 {save_path}")

    plt.show()

if __name__ == "__main__":
    # 文件路径可以根据需要更改，添加保存路径 save_path为None则不保存
    read_csv_file('datasets\\train\\squat\\squat_data_4.csv', save_path='./data/squat_data_4_plot.png')  # 保存为 PNG 格式
