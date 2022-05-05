# 主成份分析：PCA(principal Component Analysis)
import numpy as np
# import pandas as pd
from ply.cpp import xrange
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

from Util.set_console_max import set_console

base_path = 'E:\WorkSpace\AriousAlgrithms\data_set'

pd = set_console()


def linear_correlation():
    """
    线性相关矩阵秩计算样例
    :return:
    """
    M = np.array([[1, 2], [2, 4]])
    # todo:计算矩阵的秩
    rank = np.linalg.matrix_rank(M, tol=None)
    print('rank:', rank)


def pca_predict():
    """
    使用了主成分分析，来降低特征的维度，提高预测结果
    :return:
    """
    digits_test = pd.read_csv(base_path + '\optdigits.tes')
    digits_train = pd.read_csv(base_path + '\optdigits.tra', header=None)
    # todo:分割训练数据的特征向量和标记
    X_digits = digits_train[np.arange(64)]
    y_digits = digits_train[64]
    # todo:从sklearn中导入PCA
    estimator = PCA(n_components=20)
    X_pca = estimator.fit_transform(X_digits)
    # todo:初始化PCA，可以将64个维度压缩至20个维度
    # TODO:显示10类手写体数字图片经PCA压缩后的2维空间分布(其实就是画图了)
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in xrange(len(colors)):
        px = X_pca[:, 0][y_digits.values == i]
        py = X_pca[:, 0][y_digits.values == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()


def first_svr():
    """
    使用原始像素特征和经PCA压缩重建的低维特征，在相同配置的支持向量机（分类）模型上分别进行图像识别
    :return:
    """
    digits_train = pd.read_csv(base_path + '\optdigits.tra', header=None)
    digits_test = pd.read_csv(base_path + '\optdigits.tes', header=None)
    # todo:对训练数据和测试数据进行特征向量（图片像素）与分类目标的分割
    X_train = digits_train[np.arange(64)]
    y_train = digits_train[64]
    X_test = digits_test[np.arange(64)]
    y_test = digits_test[64]
    # todo:导入基于线性核的支持向量机分类器
    svc = LinearSVC()
    # todo:使用默认配置的linearSVC(),训练数据进行建模，和测试
    svc.fit(X_train, y_train)
    svc_predict = svc.predict(X_test)
    # TODO:使用PCA将原来64个维度数据压缩为20个维度
    estimator = PCA(n_components=30)
    # todo:利用训练特征决定20个正交维度的方向，并转换为原训练特征
    pca_X_train = estimator.fit_transform(X_train)
    # print('转换数据之前：', X_train)
    # print('pca_X_train,转化之后的值：', pca_X_train)  # 这里是将fit_transform的值转换为一个二位矩阵了
    # todo:将测试数据的特征也按照上面做
    pca_X_test = estimator.transform(X_test)
    # todo:再次使用默认配置的linearSVC()进行建模，然后预测结果。
    pva_svc = LinearSVC()
    pva_svc.fit(pca_X_train, y_train)
    pca_predict = pva_svc.predict(pca_X_test)
    # TODO：查看预测结果
    print('不使用PCA降维的预测结果', svc.score(X_test, y_test))
    print('使用PCA降维的预测结果', pva_svc.score(pca_X_test, y_test))


if __name__ == '__main__':
    first_svr()
"""
总结：
这里有两个方法，
    1、是画图，引用的是手写体数字图像的数据集，这里是将数据进行了降维，
        然后进行的画图，但其实中间还是有一个不太理解的，就是他画出来的是一条斜线。
    2、也是使用的手写体图像数据集，这里是使用PCA主成分分析进行了数据降维，然后进行预测的。
        这里的降维的参数，我特地是调了一下，
        1：0.2
        20：0.92
        30：0.93
        所以说这个参数，其实都需要试一试，不然的话，不太清楚做好的特征选取是哪一种。
"""
