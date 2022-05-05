# 支持向量机的回归预测，这里书本上是采用了三种不同的配制，来建立模型，而我将尝试使用更多的配置信息，来建模，然后比对他们的预测分数
from skimage.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from Util.set_console_max import set_console


def support_SVR():
    # 还是采用boston房价预测的数据集
    # todo:导入数据集
    boston = load_boston()
    # print(boston)
    # todo:数据集拆分
    X = boston.data
    y = boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    # todo:要进行特征抽取
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)
    # todo：重置一下维度
    # y_train = y_train.reshape(-1,1)
    # y_test = y_train.reshape(-1,1)

    y_train = ss_y.fit_transform(y_train.reshape(-1,1))
    y_test = ss_y.transform(y_test.reshape(-1,1))
    # TODO:向量机模型初始化，并训练数据，之后进行预测
    # todo:使用线性核函数配置的支持向量机
    linear_svr = SVR(kernel='linear')
    linear_svr.fit(X_train, y_train)
    linear_svr_predict = linear_svr.predict(X_test)
    # todo:使用多项式函数配置
    poly_svr = SVR(kernel='poly')
    poly_svr.fit(X_train,y_train)
    ploy_svr_predict = poly_svr.predict(X_test)
    # todo:使用径向基核函数
    rbf_svr = SVR(kernel='rbf')
    rbf_svr.fit(X_train,y_train)
    rbf_svr_predict = rbf_svr.predict(X_test)
    # todo:查看预测结果
    print('linear_svr_predict', linear_svr.score(X_test, y_test))
    print('poly_svr_predict',poly_svr.score(X_test,y_test))
    print('rbf_svr_predict',rbf_svr.score(X_test,y_test))
    # ?下面会有一个问题，目前就先将他注释了
    # print('The mean squared error of linear SVR is:',
    #       mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict)))
    # print('The mean absolute error of linear SVR is:',
    #       mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict)))


if __name__ == '__main__':
    support_SVR()
