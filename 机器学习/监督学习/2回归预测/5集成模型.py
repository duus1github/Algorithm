# 集成模型：这里是使用了极端随机森林Extremely Randomized Trees、普通随机森林RandomForestRegressor、随机梯度上升GradientBoostingRegressor
from Common.boston_data import boston_data_init
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor


def integrated_model():
    """
    集成模型
    :return:
    """
    X_train, X_test, y_train, y_test, ss_x, ss_y = boston_data_init('else')
    # TODO: 模型类引入，并训练数据
    # todo:随机森林训练模型
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    rfr_predict = rfr.predict(X_test)
    # todo:随机梯度上升训练模型
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train,y_train)
    gbr_predict = gbr.predict(X_train)
    # todo:极端随机森林训练模型
    etr = ExtraTreesRegressor()
    etr.fit(X_train, y_train)
    etr_predict = etr.predict(X_test)
    # TODO:查看预测结果
    print('随机森林训练模型预测结果：', rfr.score(X_test, y_test))
    print('随机梯度上升训练模型：', gbr.score(X_test, y_test))
    print('极端随机森林训练模型：', etr.score(X_test, y_test))


if __name__ == '__main__':
    integrated_model()
