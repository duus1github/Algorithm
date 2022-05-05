# 这里是将上述的分类学习算法进行集成，其实也就是说，使用多个算法进行建模，以达到最优结果
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

base_path = 'E:\WorkSpace\AriousAlgrithms\data_set'


def ensemble():
    # TODO:使用pd读取csv数据
    data_path = base_path + '\Titanic_dataSets.csv'
    data = pd.read_csv(data_path)
    print(data.info())
    # 特征的选择
    X = data[['Pclass_1', 'Pclass_2', 'Pclass_3', 'Age', 'Sex']]
    y = ['Survived']
    print(X.info())
    # 这里如果数据不完整，就是需要对数据重补充，然后在次探查数据
    # TODO；数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, data[['Survived']], test_size=0.25, random_state=33)
    # TODO:特征的抽取
    dv = DictVectorizer()
    X_train = dv.fit_transform(X_train.to_dict(orient='record'))
    X_test = dv.fit_transform(X_test.to_dict(orient='record'))
    # TODO;使用单一决策树进行模型训练和预测分析
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train,y_train)
    dtc_y_predict = dtc.predict(X_test)
    # TODO:使用随机森林分类器进行集成模型的训练以及预测分析
    rfc = RandomForestClassifier()
    # print('X_train,y_train:',X_train,y_train)
    rfc.fit(X_train,y_train)
    rfc_y_predict = rfc.predict(X_test)
    # TODO：使用梯度提升决策树进行集成模型的训练和预测分析
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train,y_train)
    gbc_y_predict = gbc.predict(X_test)
    # TODO；预测结果分析
    # todo:单一决策树预测结果分析
    print('单一决策树预测结果分析:',dtc.score(X_test,y_test))
    print("单一决策树预测结果分析 classification_report:\n",
          classification_report(y_test, dtc_y_predict, target_names=['died', 'survived']))
    # todo:随机森林分类器预测结果分析
    print('随机森林分类器预测结果分析:',rfc.score(X_test,y_test))
    print("随机森林分类器预测结果分析 classification_report:\n",
          classification_report(y_test, rfc_y_predict, target_names=['died', 'survived']))
    # todo:梯度提出决策树预测结果分析
    print('随机森林分类器预测结果分析:', gbc.score(X_test, y_test))
    print("随机森林分类器预测结果分析 classification_report:\n",
          classification_report(y_test, gbc_y_predict, target_names=['died', 'survived']))

if __name__=='__main__':
    ensemble()