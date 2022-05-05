#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:layers.py
@time:2022/04/30
"""
import math

import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    简单的GCN层，类似于https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        随机重新设置偏置参数
        :return:
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """正向传播"""
        support = torch.mm(input, self.weight)
        out = torch.spmm(adj, support)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

    def __repr__(self):
        return self.__class__.__name__ + str(self.in_features) + str(self.out_features)
