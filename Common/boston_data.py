# 因为就是说后期的学习经常使用到boston 房价的数据，所以这里就将这一块的内容单独拿了出来。
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def boston_data_init(reg_type):
    boston = load_boston()
    X = boston.data
    y = boston.target
    # todo:数据拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    if reg_type != 'tree':  # 就表明不需要特征抽取
        # todo:特征抽取
        ss_x = StandardScaler()
        ss_y = StandardScaler()
        X_train = ss_x.fit_transform(X_train)
        X_test = ss_x.transform(X_test)
        return X_train, X_test, y_train, y_test, ss_x, ss_y
    return X_train, X_test, y_train, y_test
