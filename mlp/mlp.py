import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import init
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim
def get_training_dataset(sigma=0.01):
    mu = np.array([[2, 3]])
    Sigma = np.array([[1, 0], [0, 1]])
    Sigma2 = sigma*Sigma
    positive = np.dot(np.random.randn(200, 2), Sigma2) + mu
    avg = np.array([[5, 6]])
    print(np.dot(np.random.randn(800, 2), Sigma2) + avg)
    negetive = list(np.dot(np.random.randn(800, 2), Sigma2) + avg)
    input_vecs = np.concatenate((positive,negetive),axis=0)
    # 期望的输出列表
    labels = np.zeros(1000,dtype=int)
    labels[:200] = np.ones(200,dtype=int)
    labels[200:] = np.zeros(800,dtype=int)
    return input_vecs, labels

def sepratedata(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    x_test,x_val,y_test,y_val, = train_test_split(x_test, y_test, test_size=0.5)
    return x_train, x_test,x_val,y_train, y_test,y_val
x,y = get_training_dataset()
x0,x1,x2,y0,y1,y2 = sepratedata(x,y)
x0, y0 = torch.tensor(x0.astype('float32')), torch.tensor(y0.astype('float32'))
x1, y1 = torch.tensor(x1.astype('float32')), torch.tensor(y1.astype('float32'))
x2, y2 = torch.tensor(x2.astype('float32')), torch.tensor(y2.astype('float32'))

num_inputs, num_hiddens, num_outputs = 2, 2, 1


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.hidden = nn.Linear(num_inputs, num_hiddens)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(num_hiddens, num_outputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

net = Classification()

init.normal_(net.hidden.weight, mean=0, std=0.01)
init.normal_(net.output.weight, mean=0, std=0.01)
init.constant_(net.hidden.bias, val=0)
init.constant_(net.output.bias, val=0)

loss = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

def evaluate_accuracy(x,y,net):
    out = net(x)
    sum = 0
    for index,i in enumerate(out):
        if(i.ge(0.5) ==y[index]):
            sum+=1
    a = sum/600
    return a

def evaluate_accuracytest(x,y,net):
    out = net(x)
    sum = 0
    for index,i in enumerate(out):
        if(i.ge(0.5) ==y[index]):
            sum+=1
    print("====",sum)
    a = sum/200
    return a,out

def train(net,train_x,train_y,loss,num_epochs,optimizer=None):

    for epoch in range(num_epochs):
        out = net(train_x)
        l = loss(out, train_y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_loss = l.item()

        if (epoch + 1) % 100 == 0:
            train_acc = evaluate_accuracy(train_x,train_y, net)
            print("======",train_acc)
            print('epoch %d ,loss %.4f'%(epoch + 1,train_loss)+', train acc {:.2f}%'
                  .format(train_acc*100))

num_epochs = 1000
train(net, x0, y0, loss, num_epochs, optimizer)

#validate
print("validate")
train_acc,out = evaluate_accuracytest(x1,y1,net)
print("======",train_acc)
print('train acc {:.2f}%'.format(train_acc*100))

#test
print("test")
train_acc,out = evaluate_accuracytest(x2,y2,net)
print("======",train_acc)
print('train acc {:.2f}%'.format(train_acc*100))

ans = []
for index,i in enumerate(out):
    if(i.ge(0.5)):
        ans.append(1)
    else:
        ans.append(0)
#compute fpr,tpr,threshold
print(ans)
print(y2)
fpr,tpr,threshold = roc_curve(ans, y2,pos_label=1)
roc_auc = auc(fpr,tpr) ###计算auc的值
print(roc_auc)
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()