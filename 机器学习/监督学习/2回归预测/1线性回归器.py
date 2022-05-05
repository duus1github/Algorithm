# 线性回归器
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from Util.set_console_max import set_console


def linear_regressor():
    # TODO:引入波士顿房价数据
    boston = load_boston()
    print('boston_data:\n', boston.data.shape)
    # print('boston_target:\n', boston.DESCR)
    print(boston.target)
    X = boston.data[boston.target < 50.0]
    y = boston.target[boston.target < 50.0]
    # TODO:使用pandas进行数据处理
    pd = set_console()
    # print(boston['data'])

    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    # 往boston_df中加入一列数据,列名为Target
    boston_df['Target'] = pd.DataFrame(boston['target'], columns=['Target'])

    # 将数据进行排序，
    # boston_df = boston_df.corr().sort_values(by=['Target'],ascending=False)
    print(boston_df.shape)
    # 样本数据可视化
    sns.set(palette='muted', color_codes=True)
    sns.pairplot(boston_df, vars=['RM', 'Target'])
    plt.show()
    # 清洗数据

    # print(X.shape)
    # print(y.shape)
    # TODO：数据切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    # TODO：进行特征抽取，特征标准化处理
    # 特征对象实例化
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    # 对数据进行特征抽取,这里是将四个数据都进行了特征的抽取，why?
    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)
    print('y_train:', type(X_train))
    # y_train = ss_y.fit_transform(y_train)
    # y_test = ss_y.transform(y_test)
    # TODO：使用线性回归模型linearregression和随机梯度下降SGDRegressor对房价做出预测
    # 实例化模型类
    lr = LinearRegression()
    sgdr = SGDRegressor()
    # 进行模型训练
    lr.fit(X_train, y_train)
    lr_predict = lr.predict(X_test)
    sgdr.fit(X_train, y_train)
    sgdr_predict = sgdr.predict(X_test)
    # TODO：查看预测的结果
    # 这里就还是用自带的评分吧
    print('线性回归预测结果：', lr.score(X_test, y_test))
    print('随机梯度下降预测结果：', sgdr.score(X_test, y_test))
    # TODO：尝试一下画个图。
    # 将结果画图，就需要将这些数据格式转化为dataframe的格式
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    test_data = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    lr_predict = pd.DataFrame(lr_predict)
    sgdr_predict = pd.DataFrame(sgdr_predict)
    # 数据可视化
    # plt.scatter(X_train,y_train,colot='r',label='训练数据')
    plt.plot(X_train)
    plt.xlabel('平均房间数目[MEDV]', )
    plt.ylabel('以1000美元为计价单位的房价[RM]' )
    plt.title('波士顿房价预测', fontsize=20)
    # fig = plt.figure()
    # # fig.satter()
    # ax = fig.add_axes([0, 0, 1, 1])
    # # print(X_train.shape,y_train.shape)
    # ax.scatter(train_data,color='b',label='train_data')
    # ax.scatter(test_data,color='r',lable='test_data')
    # ax.set_xlabel('RM')
    # ax.set_ylabel('Target')
    # ax.set_title('boston')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    linear_regressor()
