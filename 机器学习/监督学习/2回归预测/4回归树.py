# 回归树的预测
from Common.boston_data import boston_data_init
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def reg_tree():
    X_train, X_test, y_train, y_test = boston_data_init('tree')
    # todo:直接导入回归树
    dtr = DecisionTreeRegressor()
    # 直接进行训练即可
    dtr.fit(X_train, y_train)
    dtr_predict = dtr.predict(X_test)
    # todo:查看预测结果

    print('查看预测结果得分', dtr.score(X_test, y_test))
    # TODO:我是不是也是可以尝试画图呢
    # todo:创建画布
    plt.figure()
    # todo:绘制散点图
    print(X_train.shape,y_train.shape)
    print(len(X_train),len(y_train))
    # plt.scatter(X_train,y_train)
    plt.plot(X_test,dtr_predict)
    plt.show()


if __name__ == '__main__':
    reg_tree()
