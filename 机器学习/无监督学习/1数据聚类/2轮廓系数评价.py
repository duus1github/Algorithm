# 这个是一个轮廓系数来评价数据的结果
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def contour_coefficient():
    # todo:分割出3*2=6个子图，并再1号作图
    plt.subplot(3, 2, 1)
    # todo:初始化数据点
    x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
    x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    # todo:再1号子图做出原始数据点阵的分布
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.title('Instances')
    plt.scatter(x1, x2)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
    clusters = [2, 3, 4, 5, 8]
    subplot_counter = 1
    sc_scores = []
    for t in clusters:
        subplot_counter += 1
        plt.subplot(3, 2, subplot_counter)
        kmeans_model = KMeans(n_clusters=t).fit(X)
        for i, l in enumerate(kmeans_model.labels_):
            plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
        sc_scores.append(sc_score)
        # todo:绘制轮廓系数与不同类簇数量的直观显示图
        plt.title('K=%s,silhoouette coefficient=%0.03f' % (t, sc_score))
    plt.show()
    plt.figure()
    plt.plot(clusters, sc_scores, '*-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient Score')
    # todo:绘制轮廓系数与不同类簇数量的关系曲线
    plt.show()


def elbow_observation():
    """
    肘部观察法，
    :return:
    """
    cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
    cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
    cluster3 = np.random.uniform(3.0, 4.0, (2, 10))
    X = np.hstack((cluster1, cluster2, cluster3)).T
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    # todo:测试9钟不同聚类中心数量下，每种情况的聚类质量，并作图
    K = range(1, 10)
    meadistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        meadistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    plt.plot(K, meadistortions, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Average Dispersion')
    plt.title('Selecting k with the Elbow Method')
    plt.show()


if __name__ == '__main__':
    elbow_observation()
