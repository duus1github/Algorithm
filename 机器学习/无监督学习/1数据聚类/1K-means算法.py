# K-means算法有叫K均值算法，最经典，而且是最容易理解的模型。这里采用的数据是手写体数字图像数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

base_path = 'E:\WorkSpace\AriousAlgrithms\data_set'


def k_means_learning():
    # TODO:读取训练数据集和测试数据集
    digits = load_digits()
    # todo:这里，我看到它是有从一个链接中直接读取的数据，
    #  但我的想法是，它为什么要这样将训练数据和测试数据分开读取呢？我可以用我自己自己之前的方法来拆分么？试一试。
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)

    # TODO：从训练和测试数据集上分离出64个维度的像素特征与1个维度的数字目标
    # X_train = X_train[np.arange(64)]
    # y_train = y_train[64]
    # X_test = X_test[np.arange(64)]
    # y_test = y_test[64]
    print('看看数据：', y_test.shape, y_train.shape)
    # TODO：导入kmeans模型，并初始化，设置聚类中心数量为10
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X_train)
    # TODO：逐条判断每个测试图像所属的聚类中心。
    km_predict = kmeans.predict(X_test)
    # TODO：对性能进行评估
    # todo:使用ARI指标查看性能
    print('看看数据：', y_test.shape, km_predict.shape)
    print(metrics.adjusted_rand_score(y_test, km_predict))  # 使用数据拆分时这里得到了一个错误，查看原因与上面的数据源有关
    # TODO;这里是想画个图，来看看k的最优取值是多少
    score_list = []
    ks = []
    for i in range(5, 100):
        km = KMeans(n_clusters=i)
        km.fit(X_test, y_test)
        # km_predict = km.predict(y_test)
        score = km.score(X_test, y_test)
        score_list.append(score)
        ks.append(i)
    # todo:使用numpy转化数据为np_list类型
    arr_score = np.array(score_list)
    arr_ks = np.array(ks)
    # todo:进行画图
    fig = plt.figure()
    plt.plot(arr_score, arr_ks)
    plt.xlabel('k的取值')
    plt.ylabel('预测得分')
    plt.show()


"""
总结，画出图之后是一条平滑上升的曲线，可能是自己有个地方出错了，或者是哪里数据有些问题。
"""


def pd_kmeans_learning():
    # TODO：使用pd读取数据信息
    digits_test = pd.read_csv(base_path + '\optdigits.tes', header=None)
    digits_train = pd.read_csv(base_path + '\optdigits.tra', header=None)
    # TODO:从训练和测试数据集上都分离出64维度的像素特征与1维度的数字目标
    X_train = digits_train[np.arange(64)]
    y_train = digits_train[64]
    X_test = digits_test[np.arange(64)]
    y_test = digits_test[64]
    # TODO:导入kmeans进行训练和数据的预测
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X_train)
    km_predict = kmeans.predict(X_test)
    # TODO:这里我再调用RPI,来查看预测结果集

    print(metrics.adjusted_rand_score(y_test, km_predict))


if __name__ == '__main__':
    # pd_kmeans_learning()  # 0.6677731542678766
    k_means_learning()  # 0.6199396887805012
    # 上面这两个预测结果的区别，通过对比可以很好的看到。
    # 进行了数据64个维度像素特征分离和1维度数字目标的性能明显优于上面没有进行处理的
    # 那是不是说，数据集经过一些特征的处理，就可以使得数据预测结果变好呢？
    # 那这个特征的处理，是不是就应该依据这个数据集的结构特性呢？
