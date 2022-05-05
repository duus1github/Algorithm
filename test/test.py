# 这里test就是用来多demo，来帮助我更深入额理解一些方法的
import numpy as np
import numpy as py
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def demo_train_test_split():
    # 用于理解random_state参数
    x = np.linspace(1, 8, 8).reshape(4, 2)
    print(x)
    label = list([1, 0, 1, 1])
    # todo:使用train_test_split()方法进行拆分
    x_train, x_test, y_train, y_test = train_test_split(x, label, test_size=0.25, shuffle=True, random_state=2)
    print("x_train", x_train)
    print("x_test", x_test)


def draw_right_angle():
    """
    就是画个直角的坐标东西
    :return:
    """
    x = np.arange(1, 5)
    y = x + 1
    y2 = 3 * (x + 1)
    plt.figure()
    plt.plot(x, y, color='red')
    plt.plot(x, y2, color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def test_pcolotmesh():
    np.random.seed(19680801)
    Z = np.random.rand(6, 10)
    x = np.arange(-0.5, 10, 1)  # len = 11
    y = np.arange(4.5, 11, 1)
    print(Z)
    print(x)
    print(y)
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, Z)
    plt.show()

if __name__ == '__main__':
    test_pcolotmesh()
