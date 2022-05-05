# 分类学习的支持向量机算法的学习
"""
这是一个良/恶性乳腺癌肿瘤预测的数据集，使用支持向量机来预测。
"""
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class Classification_report:
    pass


def support_vector_machine():
    """支持向量机算法"""
    # TODO：获取数据集——>数据的清洗
    digits = load_digits()
    print('digits:', digits.DESCR)  # 这里应该注意的是，这是一个对象，所以是不能狗直接.shape的，而是需要.data之后

    # TODO: 数据分割，同样是使用model_selection中的train_test_spilt()方法
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
    print('y_train:', y_train.shape)
    print('y_test:', y_test.shape)
    # TODO：对数据进行标准化
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    # TODO: 冲sklearn.svm中导入基于线性假设的支持向量机分类其LinearSVC.,进行模型训练
    lsvc = LinearSVC()
    lsvc.fit(X_train, y_train)
    # 预测结果
    y_predict = lsvc.predict(X_test)
    # TODO：还是使用classification_report模块对预测结果进行分析。
    # 这里是使用自带的评估函数
    print('Accuracy of  Linear SVC is\n ,', lsvc.score(X_test, y_test))
    # 下面是使用classification_report，
    print("Accuracy of Classification_report :\n",
          classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))


if __name__ == "__main__":
    support_vector_machine()
