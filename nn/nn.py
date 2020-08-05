import torch
from torch.autograd import Variable
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import numpy as np
def kl(my_data):
    my_data_mean = my_data.mean()
    new_data = my_data - my_data_mean
    C = np.cov(new_data)
    c,p = np.linalg.eig(C)
    return c

data= pd.read_csv("forestfires.csv")
nums = 1000
print(data)
train_y = data.values[:,12:]
train_x = data.values[:,:12]
month = list(calendar.month_abbr)
month = [x.lower() for x in month]
week = ['','mon','tue','wed','thu','fri','sat','sun']
print(week)
print(month)
for num in range(train_x.shape[0]):
    train_x[num][2:3] = month.index(train_x[num][2:3])
for num in range(train_x.shape[0]):
    train_x[num][3:4] = week.index(train_x[num][3:4])
train_x = normalize(train_x)
train_y = normalize(train_y)
# 将tensor置入Variable中
x, y = torch.tensor(train_x.astype('float32')), torch.tensor(train_y.astype('float32'))

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 定义一个构建神经网络的类
class Net(torch.nn.Module):  # 继承torch.nn.Module类
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 获得Net类的超类（父类）的构造方法
        # 定义神经网络的每层结构形式
        # 各个层的信息都是Net类对象的属性
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    # 将各层的神经元搭建成完整的神经网络的前向通路
    def forward(self, x):
        x = F.relu(self.hidden(x))  # 对隐藏层的输出进行relu激活
        x = self.predict(x)
        return x

    # 定义神经网络


net = Net(12, 12, 1)
print(net)  # 打印输出net的结构

# 定义优化器和损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  # 传入网络参数和学习率
loss_function = torch.nn.MSELoss()  # 最小均方误差

# 神经网络训练过程
  # 动态学习过程展示
y1 = []
for t in range(3000):
    prediction = net(x)  # 把数据x喂给net，输出预测值
    loss = loss_function(prediction, y)  # 计算两者的误差，要注意两个参数的顺序
    y1.append(loss.item())
    optimizer.zero_grad()  # 清空上一步的更新参数值
    loss.backward()  # 误差反相传播，计算新的更新参数值
    optimizer.step()  # 将计算得到的更新值赋给net.parameters()
    plt.plot(t,loss.item())
    # 可视化训练过程
    if (t + 1) % 10 == 0:
        print(loss.item())


x1 = [i for i in range(3000)]
plt.plot(x1, y1, color='red', linewidth=1.0, linestyle='--')
# 显示图表
plt.show()
