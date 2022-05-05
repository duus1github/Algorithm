#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:model.py
@time:2022/05/02
@desc:这是SENet的代码层面，具体的笔记我记录在了onenote上面了
"""
from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(  # sequential():一个时序容器，会以他们传入的顺序被添加到容器中，
            nn.Linear(channel, channel // reduction, bias=False),  # 全连接层
            nn.ReLU(inplace=True),  # relu激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 全连接层
            nn.Sigmoid()  # sigmoid激活函数
        )

    def forward(self, x):
        """前向传播"""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
