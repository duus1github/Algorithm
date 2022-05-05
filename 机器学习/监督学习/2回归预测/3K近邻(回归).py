# 模型介绍
from sklearn.neighbors import KNeighborsRegressor

from Common.boston_data import boston_data_init


def k_nears():
    # todo:初始化数据
    X_train, X_test, y_train, y_test, ss_x, ss_y = boston_data_init('no')
    # todo:初始化k近邻回归器，并且调整预测方式为平均回归。
    uni_knr = KNeighborsRegressor(weights='uniform',n_neighbors=100)
    uni_knr.fit(X_train, y_train)
    uni_knr_predict = uni_knr.predict(X_test)
    # todo:调整预测方式为根据距离加权回归
    dis_knr = KNeighborsRegressor(weights='distance',n_neighbors=7)
    dis_knr.fit(X_train, y_train)
    dis_knr_predict = dis_knr.predict(X_test)
    # todo:查看预测结果数据
    print('uni_knr_predict查看预测结果：', uni_knr.score(X_test, y_test))
    print('dis_knr_predict查看预测结果：', dis_knr.score(X_test, y_test))


if __name__ == '__main__':
    k_nears()
