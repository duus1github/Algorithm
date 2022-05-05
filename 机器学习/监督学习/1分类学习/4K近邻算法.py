# K近邻算法
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def k_nearest_neighbor():
    # TODO;读取数据集
    iris = load_iris()
    # 查看一下数据信息
    print('iris 数据信息', iris.data.shape)
    print('iris 数据说明', iris.DESCR)
    # TODO：数据分割
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
    # TODO:特征提取
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    # TODO: 直接使用k近邻分类器，对数据进行类别预测
    knc = KNeighborsClassifier(n_neighbors=12)
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)
    print('y_predict:', y_predict)
    # TODO：预测性能的评估
    print('the Accuracy of K-Nearest is:', knc.score(X_test, y_test))
    print('use the classification_report to show accuracy of k-Nearest is:\n',
          classification_report(y_test, y_predict, target_names=iris.target_names))
    # TODO:这里是优化一下，我会画个图，来具体显示一下，当k位多少时，预测结果时最好的
    i_list = []
    score_list = []
    for i in range(1, 50,2):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        score_list.append(score)
        i_list.append(i)
    # todo:通过画图，来看K的最优取值
    fig = plt.Figure(dpi=100, figsize=(2, 50))
    # todo:使用numpy转换为list，因为这里面有好的方法，可以直接得到最大值
    i_arr = np.array(i_list)
    score_arr = np.array(score_list)
    plt.plot(i_arr, score_arr)
    plt.grid(True,linestyle='--')
    plt.xlabel('i_list')
    plt.ylabel('score')
    plt.show()
    # todo:画出图像之后，查看最有k是多少？
    print('最大值的k下标：', score_arr.argsort())


if __name__ == '__main__':
    k_nearest_neighbor()
