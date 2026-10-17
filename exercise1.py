import numpy as np

# 1. 赋值，选x=2，y=3
x = np.array(2.0)
y = np.array(3.0)

# =====正向计算=====
a = x * y
b = x ** 2
c = a + b
f = c ** 2
print("正向计算 f =", f)

# =====反向：手动求导，从后往前=====
df_dc = 2 * c
dc_da = 1
dc_db = 1

df_da = df_dc * dc_da
df_db = df_dc * dc_db

da_dx = y
da_dy = x
db_dx = 2 * x

# 链式合并
df_dy = df_da * da_dy
df_dx = df_da * da_dx + df_db * db_dx

print("手动numpy梯度 df/dx =", df_dx)
print("手动numpy梯度 df/dy =", df_dy)





import torch

# 创建tensor，打开requires_grad记录计算图
x_t = torch.tensor(2.0, requires_grad=True)
y_t = torch.tensor(3.0, requires_grad=True)

# 正向计算
a_t = x_t * y_t
b_t = x_t ** 2
c_t = a_t + b_t
f_t = c_t ** 2

# 反向传播
f_t.backward()

print("torch自动微分 df/dx =", x_t.grad)
print("torch自动微分 df/dy =", y_t.grad)
