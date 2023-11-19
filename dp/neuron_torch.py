import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class OurNeuralNetwork(nn.Module):
    def __init__(self):
        # 继承nn.Module
        super(OurNeuralNetwork, self).__init__()
        # 全连接隐藏层 2x2的
        self.hidden_layer = nn.Linear(2, 2)
        # 输出层
        self.output_layer = nn.Linear(2, 1)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一个完全连接层，然后使用sigmoid
        h = self.sigmoid(self.hidden_layer(x))
        # 输出层，然后使用sigmoid
        o = self.sigmoid(self.output_layer(h))
        return o

# 定义数据集 data
data = torch.tensor([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6]
], dtype=torch.float32)
all_y_trues = torch.tensor([
    [1],
    [0],
    [0],
    [1]
], dtype=torch.float32)

# 训练神经网络!
network = OurNeuralNetwork()
# 设置loss
loss_function = nn.MSELoss()
# 随机梯度下降，学习率为0.01
optimizer = optim.SGD(network.parameters(), lr=0.01)
# loss存为一个list
loss_list = []

# epochs设为1000
epochs = 1000
for epoch in range(epochs):
    #梯度归零
    optimizer.zero_grad()
    #传递数据
    predictions = network(data)
    #MSE损失
    loss = loss_function(predictions, all_y_trues)
    #反向传播计算
    loss.backward()
    #根据步长和梯度反方向迭代优化
    optimizer.step()
# 每10轮存一组loss
    if epoch % 10 == 0:
        print(f"Epoch {epoch} loss: {loss.item()}")
        loss_list.append(loss.item())

# matplotlib绘图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Neural Network Loss vs. Epochs")
# x为epoch，y为loss
epochs_range = range(0, epochs, 10)
plt.plot(epochs_range, loss_list)
# 可视化
plt.show()