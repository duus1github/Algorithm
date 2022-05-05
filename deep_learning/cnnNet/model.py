#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:model.py,这里是定义手写数字体的cnn神经网络模型的
@time:2022/04/28
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # todo:1、输入图片的channel,6，5x5
        self.conv1 = nn.Conv2d(1, 6, 5)  # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # todo:y=wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # TODO:定义前向传播
    def forward(self, x):
        # todo:两个池化层
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 第一层池化
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 第二层池化
        x = torch.flatten(x, 1)  # 展平除批次维度之外的所有维度

        # todo:两层激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = Net()
    print('net:\n', net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())
    print('--------------')
    print('输出：')
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print('out:', out)
    # todo:把梯度清零后
    net.zero_grad()
    out.backward(torch.randn(1,10))
