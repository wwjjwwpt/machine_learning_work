import torch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import calendar
input_size = 12
output_size = 1
learning_rate = 0.000001
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
class linear_regression(nn.Module):
    """

     linear regression module

    """
    def __init__(self):
        super(linear_regression, self).__init__()
        # Linear函数只有一个输入一个输出
        self.linear = nn.Linear(12, 1)

    def forward(self, x):
        out = self.linear(x)
        return out
model = linear_regression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(nums):



    # 将数据转换成tensor变量，这样可以自动求导

    inputs = torch.tensor(train_x.astype('float32'))
    targets = torch.tensor(train_y.astype('float32'))





    # 前向 + 后向 + 优化

    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, targets)

    loss.backward()

    optimizer.step()



    if epoch % 20 == 0:

        print(" current loss is %.5f" % loss.item())










