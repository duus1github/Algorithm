# 这里是线性回归的一个案例，采用的是
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from Util.set_console_max import set_console

base_path = 'E:\WorkSpace\AriousAlgrithms'


def linear():
    # TODO:1、读取数据并初始化
    column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell size', 'Bare Nuclei', ' Bland Chromatin',
                    'Normal Nucleoli', 'Mitoses', 'class']
    # 使用pandas读取数据
    path = base_path + '\data_set\\breast-cancer-wisconsin.data'
    pd = set_console()
    data = pd.read_csv(path, names=column_names)
    # print(data)
    # todo:将缺失值替换为nan
    data = data.replace(to_replace='?', value=np.nan)
    # todo:丢弃带有缺失值的数据
    data = data.dropna(how='any')


    # todo:输出数据量
    print('查看一下数据集 ')
    print(data.shape)
    print('data:20:',data.head(20))

    # TODO:2、训练和测试数据
    # todo:先将数据分割,这里使用的是sklearn里面的train_test_split()模块来进行分割的
    # todo:随机采样25%的数据用于测试，剩下75%的数据用于构建训练模型
    # ?这里我第一次的时候是遇到了问题，主要愿意就还是上一行代码的data.shape导致的，其实简单的话就是将这行代码注释掉就可以了
    X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
                                                        test_size=0.25, random_state=33)

    # todo:查验训练样本的数量和类别分布
    print("y_train", y_train.value_counts(), )
    # todo:查验测试样本
    print("x_train", y_test.value_counts(), )
    # TODO:3、使用线性分类模型从事良/恶性肿瘤预测任务
    # todo:标准化数据保证每个维度的特征数据方差为1，均值为0，
    ss = StandardScaler()
    # !标准化中fit_transform()与transform()区别
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    # todo:初始化LogisticRegression与SGDClassifier
    lr = LogisticRegression()
    sgdc = SGDClassifier()
    # todo:调用fit函数/模块用来训练模型
    lr.fit(X_train, y_train)
    # todo:使用训练好的模型对x_test进行预测，
    lr_y_predict = lr.predict(X_test)
    # todo:调用SGDClassifier中的fit来训练模型
    sgdc.fit(X_train, y_train)

    # todo:同样进行x_test数据的预测
    sgdc_y_predict = sgdc.predict(X_test)
    # TODO:4、使用线性分类模型从事良/恶性肿瘤预测任务
    # todo:使用逻辑回归模型自带的评分函数score获得模型在测试集上的准确性结果
    print('Accuracy of LR Classifier:', lr.score(X_test, y_test))
    # todo:使用classification_report()模块来获得其他三个模块的结果
    print('使用classification_report:\n',
          classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))
    # todo:使用随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果。
    print('Accuracy of SGD Classifier:', sgdc.score(X_test, y_test))
    # todo:使用classification_report()模块来获得其他三个模块的结果
    print('使用classification_report:\n',
          classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))
    # TODO:使用matplotlib 来画出图
    plt.figure()
    print('查看长度：')
    print(len(X_train),X_train)
    print(len(y_train),y_train)
    # plt.scatter(X_train, y_train, color='green')
    # plt.plot(X_train, lr_y_predict, color='black', linewidth=4)
    # plt.title('Training data')
    # plt.show()


if __name__ == "__main__":
    linear()
