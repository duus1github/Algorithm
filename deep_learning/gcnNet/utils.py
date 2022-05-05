#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:duways
@file:utils.py
@time:2022/04/29
"""
import numpy as np
import scipy.sparse as sp
import torch
from numpy import identity


def encode_onehot(labels):
    classes = set(labels)
    # todo:下面这个用法很巧妙，要学习一下
    classes_dict = {c: identity(len(classes))[i, :] for i, c in enumerate(classes)}  # c:是具体的值，i:为index
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


# path = './data/cora'
def load_data(path='./data/cora/', dataset='cora'):
    print('Loading {} dataset...'.format(dataset))
    # todo:获取特征的标签
    idx_feature_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    feature = sp.csr_matrix(idx_feature_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_feature_labels[:, -1])
    # todo:创建图
    idx = np.array(idx_feature_labels[:, 0], dtype=np.int32)  # 节点
    idx_map = {j: i for i, j in enumerate(idx)}  # 构建节点的索引字典
    # todo:导入edges的数据
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # todo:构建的邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)  # coo_matrix:座标格式（即三维格式）
    # todo:建立对称邻接矩阵，计算转置矩阵，将有向图，转成无向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # multiply():对两个数组的对应元素进行计算，
    # todo:归一化,使用自行定义的归一化方法
    feature = normalize(feature)
    adj = normalize(adj + sp.eye(adj.shape[0]))  # 对A+I归一化
    # 训练，验证，测试的样本
    idx_train = range(140)
    idx_val = range(200, 400)
    idx_test = range(500, 1500)
    # 将numpy的数据转化为torch格式
    feature = torch.FloatTensor(np.array(feature.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)  # 训练集索引
    idx_val = torch.LongTensor(idx_val)  # 验证集索引
    idx_test = torch.LongTensor(idx_test)  # 测试集索引
    return adj, feature, labels, idx_train, idx_val, idx_test


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy稀疏矩阵转化为torch稀疏矩阵
    :param sparse_mx:
    :return:
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )  # from
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """行归一化稀疏矩阵"""
    rowsum = np.array(mx.sum(1))  # 矩阵求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.  # 判断将值为无穷大的值，置为0
    r_mat_inv = sp.diags(r_inv)  # 构造对角矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-*A,非对称方式，简化方式 dot():点乘，即将每个对应的点进行相乘，然后相加
    return mx


def accuracy(output, labels):
    """计算准确度"""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
