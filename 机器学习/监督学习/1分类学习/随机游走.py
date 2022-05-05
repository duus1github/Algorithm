# 这是一个关于随机游走的算法练习
# 1、单一的500步随机游走的例子，从0开始步长为1和-1
import random
import matplotlib.pyplot as plt
import numpy as np


def simple_random_walk():
    # 定义变量
    position = 0
    walk = []
    steps = 500
    for i in range(0, steps):
        step = 1 if random.randint(0, 1) else -1
        position += step
        walk.append(position)
    print(walk)
    return walk


def multiple_random_salk():
    """
    多个随机游走,
    这里的思想其实就是利用np来创建一个矩阵，二维矩阵吧，8个子矩阵(由500个数)，由0，1，-1组成,
    :return:
    """
    nsteps = 500
    nwalks = 8
    draws = np.random.randint(0, 2, size=(nwalks, nsteps))
    for draw in draws:
        print("draws:",draw)
    # print("draws:", draws.__len__())
    steps = np.where(draws > 0, 1, -1) # 是numpy中一个用于筛选数据的方法。
    walks = steps.cumsum(1)  # 这个方法可以将多个list组合成一个list
    return walks


# 开始画图了
if __name__ == "__main__":
    walk = multiple_random_salk()
    print(walk)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(8):
        ax.plot(walk[i])
    plt.show()
