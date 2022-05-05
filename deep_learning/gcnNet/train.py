#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:train.py
@time:2022/04/29
"""

from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# todo:这里是导入两个包
from deep_learning.gcnNet.model import GCN
from deep_learning.gcnNet.utils import load_data, accuracy

parse = argparse.ArgumentParser()
parse.add_argument('--no-cuda', action='store_true', default=False,
                   help='disables CUDA training')
parse.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass')
parse.add_argument('--seed', type=int, default=42, help='random seed.')
parse.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parse.add_argument('--lr', type=float, default=0.1, help='inital learning rate')
parse.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay(L2 loss on parameters)')
parse.add_argument('--hidden', type=int, default=16, help='numbers of hidden units')
parse.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1-keep probability).')

args = parse.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()  # 定义cuda，
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# TODO:加载数据
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# todo:初始的模型GCN和优化器
model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1,
            dropout=args.dropout)
# todo:初始的优化器
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()
    idx_val = idx_val.cuda()


def train(epoch):
    """训练模型"""
    t = time.time()
    model.train()  # 模型训练
    optimizer.zero_grad()  # 梯度置为0
    out = model(features, adj) # 运行模型，输入参数
    loss_train = F.nll_loss(out[idx_train], labels[idx_train])
    acc_train = accuracy(out[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if not args.fastmode:
        # 分别评估验证集的性能，再验证运行期间停用dropout
        model.eval()
        out = model(features, adj)
    loss_val = F.nll_loss(out[idx_val], labels[idx_val])
    acc_val = accuracy(out[idx_val], labels[idx_val])
    print('epoch:{:04d}'.format(epoch + 1),
          'loss_train:{:04f}'.format(loss_train.item()),
          'acc_train:{:04f}'.format(acc_train),
          'loss_val:{:04f}'.format(loss_val),
          'acc_val:{:04f}'.format(acc_val),
          'time:{:.4f}s'.format(time.time() - t),
          )


# def test():
#     """测试模型"""
#     model.eval()
#     out = model(features, adj)
#     loss_test = F.nll_loss(out[idx_test], labels[idx_test])
#     acc_test = accuracy(out[idx_test], labels[idx_test])
#     print("test set results:",
#           'loss={:.4f}'.format(loss_test.item()),
#           'accuracy={:.4f}'.format(acc_test.item()))

if __name__ == '__main__':
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print('optimizer Finished!')
    print('total time elapsed:{:.4f}s'.format(time.time() - t_total))

# test()
