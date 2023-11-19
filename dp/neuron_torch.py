import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class OurNeuralNetwork(nn.Module):
    def __init__(self):
        super(OurNeuralNetwork, self).__init__()
        self.hidden_layer = nn.Linear(2, 2)
        self.output_layer = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.sigmoid(self.hidden_layer(x))
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
loss_function = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=0.1)
loss_list = []

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = network(data)
    loss = loss_function(predictions, all_y_trues)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} loss: {loss.item()}")
        loss_list.append(loss.item())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Neural Network Loss vs. Epochs")

epochs_range = range(0, epochs, 10)
plt.plot(epochs_range, loss_list)
plt.show()