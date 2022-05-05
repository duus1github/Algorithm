#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:train.py
@time:2022/05/03
"""
import torch
from torch import nn, optim

from deep_learning.cnnNet.model import Net


def main():
    """mian,训练部分"""
    input = torch.randn(1, 1, 32, 32)
    net = Net()
    # params = list(net.parameters())
    # out = net(input)
    # todo:使用随机梯度将所有参数和反向传播的梯度缓冲区归零
    # net.zero_grad()
    # out.backward(torch.randn(1, 10))

    # todo:定义损失函数
    # out = net(input)
    target = torch.randn(10)
    target = target.view(1, -1)  # 使target与输出的形状相同
    criterion = nn.MSELoss()
    # loss = criterion(out, target)

    # print(loss.grad_fn)  # 使用这个方法，可以看到一个计算图
    # todo:反向传播误差
    # net.zero_grad()
    # loss.backward()

    # todo:更新权重
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    out = net(input)
    loss = criterion(out, target)
    loss.backward()
    print('loss\n',loss)
    optimizer.step()


if __name__ == '__main__':
    main()
