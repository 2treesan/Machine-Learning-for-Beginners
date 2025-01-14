import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.chdir('./.private')
data = pd.read_csv('data_linear.csv').values
N = data.shape[0]
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')

x = np.hstack((np.ones((N, 1)), x))
w = np.array([0., 1.]).reshape(-1, 1)

# snake_case  // camelCase

epoch = 100
loss = []
lr = 1e-6

for i in range(epoch):
    r = np.dot(x, w) - y
    loss.append(0.5 * np.sum(r * r))
    w[0] -= lr * np.sum(r)
    w[1] -= lr * np.sum(np.multiply(r, x[:, 1].reshape(-1, 1)))
    print(loss[i])

predict = np.dot(x, w)
plt.plot((x[0, 1], x[N - 1, 1]), (predict[0], predict[N - 1]), 'r', label = 'Đường hồi quy')  # Nối tất cả các điểm

# Thêm thông tin đồ thị
plt.xlabel('Mét vuông')
plt.ylabel('Giá nhà')
plt.title('Đường thẳng dự đoán giá nhà')
plt.show()

plt.plot(range(epoch), loss)
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('loss function')
plt.show()


