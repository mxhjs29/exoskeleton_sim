import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, savgol_filter, medfilt
import matplotlib.pyplot as plt
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from SMPL.src.assistance_policy import ExoNet

# ==== 1. 读取数据 ====
# 假设CSV文件包含列：theta1, dtheta1, theta2, dtheta2, tau1, tau2
data = pd.read_csv('/home/chenshuo/PycharmProjects/move_sim/SMPL/data/obs_truth/exo_im_data.csv', sep="\t")

# 输入与输出
timestamp = data['timestamp'].values
fs = 1 / np.mean(np.diff(timestamp))
X = data[['theta_l', 'dtheta_l', 'theta_r', 'dtheta_r']].values
y = data[['tau_l', 'tau_r']].values

# 滤波
def butter_lowpass_filter(x, fs, fc=8.0, order=3):
    nyq = 0.5 * fs
    wn = fc / nyq
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)

def median_despike(x, k=3):
    """简易中值滤波，k=3,5... 奇数核"""
    return medfilt(x, kernel_size=k)

cols_in = ["theta_l", "dtheta_l", "theta_r", "dtheta_r"]
cols_out = ["tau_l", "tau_r"]
for c in cols_in + cols_out:
    arr = data[c].values
    raw = arr.copy()
    arr = median_despike(arr, k=3)
    arr = butter_lowpass_filter(arr, fs, fc=8.0, order=3)
    data[c] = arr
    plt.figure(figsize=(8, 3))
    plt.plot(raw, color='r', linewidth=0.8, label='Raw')
    plt.plot(arr, color='g', linewidth=1.2, label='Filtered')
    plt.title(f'{c} (red=raw, green=filtered)')
    plt.xlabel('Sample index')
    plt.ylabel(c)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==== 2. 数据标准化 ====
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# ==== 3. 划分训练/测试集 ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 4. 转换为Tensor ====
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ==== 5. 定义神经网络 ====

model = ExoNet()

# ==== 6. 定义优化器和损失函数 ====
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== 7. 训练 ====
for epoch in range(20000): 
    optimizer.zero_grad() 
    outputs = model(X_train) 
    loss = criterion(outputs, y_train) 
    loss.backward() 
    optimizer.step() 
    if (epoch+1) % 50 == 0: 
        print(f'Epoch [{epoch+1}/20000], Loss: {loss.item():.6f}')
# ==== 8. 保存模型和标准化器 ====
torch.save(model.state_dict(), "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/assistance_policy/exo_model.pth")
print("✅ 模型已保存到 /home/chenshuo/PycharmProjects/move_sim/SMPL/data/assistance_policy/exo_model.pth")
joblib.dump(scaler_X, "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/assistance_policy/scaler_X.pkl")
joblib.dump(scaler_y, "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/assistance_policy/scaler_y.pkl")
print("✅ 标准化器已保存 /home/chenshuo/PycharmProjects/move_sim/SMPL/data/assistance_policy/(scaler_X.pkl, scaler_y.pkl)")

# ==== 9. 测试 ====
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f'Test MSE: {test_loss.item():.6f}')

# ==== 10. 反标准化预测结果 ====
y_pred_real = scaler_y.inverse_transform(y_pred.numpy())
print("Sample predicted torques:\n", y_pred_real[:5])



