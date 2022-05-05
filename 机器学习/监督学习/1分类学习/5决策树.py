# 决策树啦
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

base_path = 'E:\WorkSpace\AriousAlgrithms\data_set'


def decision_tree():
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
    # TODO：使用决策树进行训练
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train,y_train)
    y_predict = dtc.predict(X_test)
    # TODO:查看预测结果
    print('Accuracy of  Linear SVC is\n ,', dtc.score(X_test, y_test))
    # 下面是使用classification_report，
    print("Accuracy of Classification_report :\n",
          classification_report(y_test, y_predict, target_names=['died','survived']))

if __name__ == '__main__':
    decision_tree()
