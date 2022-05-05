# 朴素贝叶斯算法
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def naive_bayes():
    """朴素贝叶斯算法"""
    # TODO:从sklearn中获取新闻的数据集，然后进行清洗
    news = fetch_20newsgroups()
    # 查看一下数据集
    print('news:', len(news))
    print('first new:', news.data[0])
    # TODO:数据的分割
    X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, train_size=0.25, random_state=33)

    # TODO:特征化，这里因为是在文本中所 以要先对其进行特征提取，然后在特征化
    cv = CountVectorizer()
    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)
    # TODO:调用朴素贝叶斯实例进行模型训练
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    # TODO:测试模型
    y_predict = mnb.predict(X_test)
    # TODO:查看测试结果。
    print('函数自带评分：\n', mnb.score(X_test, y_test))
    print('查看结果：\n', classification_report(y_test, y_predict, target_names=news.target_names))
    # 我得到的结果似乎不尽人意，


if __name__ == '__main__':
    naive_bayes()
