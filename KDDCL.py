# -*- coding: utf-8 -*-
"""
Created on Fri May  7 20:25:58 2021

@author: Administrator
"""

import torch
from torch import nn
from torch.nn import init
import numpy as np
import pandas as pd
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
 
 
#如果有gpu，就调用gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 256
 
#读取KDD-cup网络安全数据,将标签数字化
df1 = pd.read_csv('./data/NSL_KDD_Dataset/KDDTrain+2.txt')
df2 = pd.read_csv('./data/NSL_KDD_Dataset/KDDTest+.txt')
df1.columns = [x for x in range(43)]
df2.columns = [x for x in range(43)]
 
#将测试集中多余的标签删去（测试集有的攻击类型在训练集中未出现，我们删除这类样本）
s1 = set(np.array(df1[41]).tolist())
df2 = df2[df2[41].isin(s1)]
df = pd.concat([df1,df2])
#42列无用，删去
del df[42]
#获取特征和标签
labels = df.iloc[:, 41]
#data = df.drop(columns=[6, 8, 10, 13,14,15,16,17,18,19,20,21,25,26,27,30,41])
data = df.drop(columns=[6,8,10,13,14,15,16,17,18,19,20,21,25,26,27,30,41])

#标签编码
le = LabelEncoder()
labels =le.fit_transform(labels).astype(np.int64)
print(le.classes_)
 
 
#特征编码
data[1] = le.fit_transform(data[1])
data[2] = le.fit_transform(data[2])
data[3] = le.fit_transform(data[3])
 
#标签和特征转成numpy数组
data = np.array(data)
labels = np.array(labels)
 
#特征值归一化
min_max_scaler = MinMaxScaler()
data = min_max_scaler.fit_transform(data)
 
#转成torch.tensor类型
data=data.reshape(-1,1,5,5)
labels = torch.from_numpy(labels)
data = torch.from_numpy(data).float()
 
x_train, x_test, y_train,y_test = data[:125972], data[125972:], labels[:125972], labels[125972:]
 
#将数据集打包成DataLoader
train_dataset = Data.TensorDataset(x_train, y_train)
train_dataset.data = train_dataset.tensors[0]
train_dataset.targets = train_dataset.tensors[1]
 
#将数据集打包成DataLoader
test_dataset = Data.TensorDataset(x_test, y_test)
test_dataset.data = test_dataset.tensors[0]
test_dataset.targets = train_dataset.tensors[1]
labels = ['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep',
 'land', 'loadmodule', 'multihop', 'neptune', 'nmap', 'normal', 'perl', 'phf',
 'pod', 'portsweep', 'rootkit' ,'satan', 'smurf', 'spy', 'teardrop',
 'warezclient', 'warezmaster']
 
train_dataset.classes = labels
test_dataset.classes = labels
 
train_dataset.classes_to_idx = {i: label  for i, label in enumerate(labels)}
test_dataset.classes_to_idx = {i: label  for i, label in enumerate(labels)}
 
 
train_iter = Data.DataLoader(train_dataset, batch_size, shuffle=True)
 
 
#感知机
#定义模型
num_inputs, num_hiddens, num_outputs = 41, 50, 23
net = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.BatchNorm2d(16),
    nn.Conv2d(16, 32, kernel_size=3, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.BatchNorm2d(32,
    nn.Linear(128, 64),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(64, 23)
).to(device)
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
#定义损失函数
 loss = torch.nn.CrossEntropyLoss()
#定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-4)
 
 
num_epochs = 25
list_acc = []
 
#训练模型
for epoch in range(1, num_epochs+1):
    
    train_l_sum, train_acc_sum, n =0.0, 0.0, 0
    test_acc_sum = 0.0
    
    for data, label in train_iter:
        #如果有gpu，就使用gpu加速
        data = data.to(device)
        label = label.to(device)
        
        output = net(data)
        
        l = loss(output, label).sum()
        
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
        train_l_sum += l.item()
        train_acc_sum += (output.argmax(dim=1) == label).sum().item()
        n += label.shape[0]
        
        
        with torch.no_grad():
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            
            output = net(x_test)
            
            test_acc_sum = (output.argmax(dim=1) == y_test).sum().item()
        
    print('epoch %d, train loss %.6f,  train acc %.3f, test acc %.3f'
          % (epoch , train_l_sum/n, train_acc_sum/n, test_acc_sum /y_test.shape[0]))
    list_acc.append(test_acc_sum/y_test.shape[0])
 
#画出精度变化图
plt.plot(range(len(list_acc)), list_acc)
plt.show()
