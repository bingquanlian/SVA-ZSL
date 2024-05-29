import torch
import torch.nn as nn
from torch.nn import functional as F

# total = 0
# for i in range(0, 7057, 128):
#     total += 1
# print(total)
#
# print(torch.arange(50))
#
#
# # 假设有5个样本和3个类别
# num_samples = 5
# num_classes = 6
#
# # 随机生成样本的预测结果和真实标签
# predictions = torch.randn(num_samples, num_classes)  # 预测结果，形状为(5, 3)
# print(predictions)
# labels = torch.tensor([1, 5, 0, 1, 2])  # 真实标签，形状为(5,)
# print(labels)
#
# # 创建交叉熵损失函数
# loss_fn = nn.CrossEntropyLoss()
#
# # 计算损失
# loss = loss_fn(predictions, labels)
# loss_fn_1 = F.cross_entropy(predictions, labels)
#
# 打印结果
# print("损失值:", loss.item())
# print("损失值:", loss_fn_1.item())
# a = 0
# b = 0
# for i in range(5):
#     a += 1
#     for j in range(5):
#         b = a * b
#         print(b)


# ntrain = 7027
# batch_size = 128
# idx = torch.randperm(ntrain)[0:batch_size]
# print(idx)


"""
# 定义一个简单的神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        x = self.fc(x)
        return x

    # 定义损失函数


criterion = nn.CrossEntropyLoss()

# 假设输入数据为一个批次，每个样本的特征向量为128维
inputs = torch.randn(32, 128)

# 假设目标标签为一个批次，每个样本的标签为128个类别中的一个
targets = torch.randint(128, size=(32,))

# 将目标标签转换为二进制编码形式
targets_binary = torch.zeros(32, 128)
for i in range(32):
    targets_binary[i, targets[i]] = 1

# 计算模型的预测值
outputs = Net()(inputs)

# 计算损失值
loss = criterion(outputs, targets_binary)
print(loss)

"""

# # 打开文件，使用 'w' 模式表示写入模式
# with open('best_accuracy.txt', 'r') as file:
#     # 将数字转换为字符串并写入文件
#     data = file.read()
#     print(data)



