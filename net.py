import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
import torch.nn as nn

# 下载数据集
training_data = datasets.MNIST(
    root='data',        # 数据下载后存储的根目录文件夹名称
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 60  # 初始化训练轮数
# 加载 mnist 数据集 : 返回 containing batch_size=10 features and labels respectively
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)  # 共60000条数据
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)       # 共10000条数据

# training_data[0] : ((1, 28, 28), label)
for X, y in test_dataloader:
    print(f"Shape of X[N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.size()}")
    break

# utils.draw_sample_image(training_data[0][0])

# 训练集是否使用 CPU 或者 GPU device 进行训练
device = "cuda" if torch.cuda.is_available else "cpu"
print(device)
device = "cpu"
# 定义神经网络 (28*28, 10)
class NeuralNetwork(nn.Module):
    """:arg
        在Pytorch中定义神经网络，我们创建 NeuralNetwork 类 继承nn.Module。我们定义在 __init__ 函数中定义网络的层数，
        在 forword() 中明确 数据怎样通过 网络。为了加速神经网络的训练，如果GPU可用就把它放到GPU上。

        init : 初始化神经网络的曾说以及训练数据要通过的方法
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 30),  # z = x·w^T + b, 参数表示：(输入层神经元个数, 输出层神经元个数)
            nn.Sigmoid(),
            nn.Linear(30, 10),
            nn.Sigmoid()
        )

    # 前向传播
    def forward(self, x):
        # print('-'*6 + 'forward' + '-'*6)
        x = self.flatten(x)                 # 把 (1, 28, 28) 转为 (1, 784)
        # print('----flatten()-----')
        # print(x.shape)
        logits = self.linear_relu_stack(x)
        return logits
    pass


model = NeuralNetwork().to(device)   # 使用 GPU 训练
print(model)


# X = torch.rand(1, 28, 28, device=device)
# print("------logits------")
# logits = model(X)   # 执行 forward() 方法
# print(logits)
# print(logits.shape)
#
# # 模型层数
#
# input_image = torch.rand(3, 28, 28)
# print(input_image.size())

# 初始化超参数(层数、每层神经元个数，训练轮数)

# 测试网络性能

# 优化模型参数
loss_fn = nn.CrossEntropyLoss()       # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=3.0)
print('---------------------------')
# for param in model.parameters():
#     print(type(param), param.size())


def train(dataloader, model, loss_fn, optimizer):
    """:arg
        dataloader : 含有 batch_size 个训练集
        model : 神经网络类的实例
        loss_fn : 损失函数
        optimizer: 优化器，使用随机梯度下降算法，反向传播误差更新权重和偏置
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # 计算激活层误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播 (Backpropagation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        pass
    pass


def test(dataloader, model, loss_fn):
    ''':arg
        We also check the model’s performance against the test dataset to ensure it is learning.
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    pass


# epochs ： 迭代次数
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

from time import time

t = time()
test(test_dataloader, model, loss_fn)
print(time() - t)

# 保存模型
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# 加载模型
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = pred[0].argmax(0), y
    print(f'Predicted: "{predicted}", Actual: "{actual}"')