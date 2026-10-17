import numpy as np
import torch
from sklearn.datasets import make_moons

# 固定随机种子，保证结果每次一样
np.random.seed(42)
torch.manual_seed(42)

# ========= 1. 取Task1 make_moons的1个样本 =========
X_all, y_all = make_moons(n_samples=100, noise=0.1)
# X:输入特征，1个样本，2维特征
# 拿第一个样本
X = X_all[0:1, :]   # shape (1, 2)，1个样本，2维特征
y = y_all[0:1]      # shape (1,)，标签0/1

# ========= 2. 初始化网络参数 =========
in_dim = 2
hidden_dim = 3
out_dim = 1

# W1：输入→隐藏层权重；b1：隐藏层偏置输入 

# W1:输入→隐藏层  (2,3)
W1 = np.random.randn(in_dim, hidden_dim) * 0.1
b1 = np.zeros((1, hidden_dim))

# W2：隐藏→输出权重；b2:输出层偏置

# W2:隐藏→输出 (3,1)
W2 = np.random.randn(hidden_dim, out_dim) * 0.1
b2 = np.zeros((1, out_dim))

print("===== 【Numpy 前向传播】 =====")
# --------正向计算--------
Z1 = np.dot(X, W1) + b1         # 第一层线性输出 shape=(1,3)
A1 = np.maximum(0, Z1)          # ReLU激活后 shape=(1,3)         
Z2 = np.dot(A1, W2) + b2        # 第二层线性输出 lpgit shape=(1,1)
y_hat = 1 / (1 + np.exp(-Z2))   # Sigmoid预测概率 shape=(1,1)
# 二元交叉熵损失
loss = -(y * np.log(y_hat) + (1-y)*np.log(1-y_hat)) # scalar 标量

print(f"Z1 shape: {Z1.shape}")
print(f"A1 shape: {A1.shape}")
print(f"Z2 shape: {Z2.shape}")
print(f"预测y_hat: {y_hat.item():.4f}")
print(f"BCE Loss: {loss.item():.4f}")

# --------反向传播：手动链式求导--------
dL_dZ2 = y_hat - y                  # (1,1)  sigmoid+bce的简化梯度
dL_dW2 = np.dot(A1.T, dL_dZ2)       # (3,1)
dL_db2 = np.sum(dL_dZ2, axis=0)     # (1,1)

dL_dA1 = np.dot(dL_dZ2, W2.T)       # (1,3)
dA1_dZ1 = (Z1 > 0).astype(float)    # ReLU导数 mask
dL_dZ1 = dL_dA1 * dA1_dZ1           # (1,3)

dL_dW1 = np.dot(X.T, dL_dZ1)        # (2,3)
dL_db1 = np.sum(dL_dZ1, axis=0)     # (1,3)

print("\n=====【Numpy手动求出梯度】 =====")
print("dL/dW1:\n", dL_dW1)
print("dL/db1:\n", dL_db1)
print("dL/dW2:\n", dL_dW2)
print("dL/db2:\n", dL_db2)

# ======================================
# ========= PyTorch Autograd验证 =========
# ======================================
print("\n=====【Pytorch自动求梯度】 =====")
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

# 参数，打开requires_grad
W1_t = torch.tensor(W1, dtype=torch.float32, requires_grad=True)
b1_t = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
W2_t = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
b2_t = torch.tensor(b2, dtype=torch.float32, requires_grad=True)

# 前向
Z1_t = X_t @ W1_t + b1_t
A1_t = torch.relu(Z1_t)
Z2_t = A1_t @ W2_t + b2_t
y_hat_t = torch.sigmoid(Z2_t)
# BCE
loss_t = -(y_t * torch.log(y_hat_t) + (1 - y_t)*torch.log(1 - y_hat_t))

# 反向
loss_t.backward()

print("torch dL/dW1:\n", W1_t.grad.numpy())
print("torch dL/db1:\n", b1_t.grad.numpy())
print("torch dL/dW2:\n", W2_t.grad.numpy())
print("torch dL/db2:\n", b2_t.grad.numpy())

# =========对比误差：numpy梯度 vs torch梯度=========
print("\n===== 梯度最大绝对误差 =====")
err_W1 = np.max(np.abs(dL_dW1 - W1_t.grad.numpy()))
err_b1 = np.max(np.abs(dL_db1 - b1_t.grad.numpy()))
err_W2 = np.max(np.abs(dL_dW2 - W2_t.grad.numpy()))
err_b2 = np.max(np.abs(dL_db2 - b2_t.grad.numpy()))
print(f"W1最大误差: {err_W1:.2e}")
print(f"b1最大误差: {err_b1:.2e}")
print(f"W2最大误差: {err_W2:.2e}")
print(f"b2最大误差: {err_b2:.2e}")
print("误差极小，说明手动推导的梯度公式正确 ")
