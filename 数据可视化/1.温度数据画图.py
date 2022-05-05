# 学习maplotlib画图的第一次练习，这里是随机生成了背景上海两个地方的温度数据画图
import numpy as np
import matplotlib.pyplot as plt
#新增加的两行,以显示matplotlib中文字体问题
import matplotlib
matplotlib.rc("font",family='YouYuan')

# 直接画吧，这里就不写函数了
# TODO:随机生成数据，借助np的random
x = [i for i in np.arange(60)]
y_beijing = [np.random.randint(10, 15) for i in x]
y_wuyuan = [np.random.randint(5, 10) for i in x]
print(x,y_wuyuan)
# TODO：创建画布
plt.figure(figsize=(20,16),dpi=100)
# TODO: 绘制折线图
plt.plot(x,y_beijing,label='北京',color='g',linestyle='-')
plt.plot(x,y_wuyuan,label = '婺源',color='r',linestyle='-.')
# todo:添加x,y的刻度
y_ticks = range(40)
x_ticks_labels = ['11点{}分'.format(i) for i in x]
plt.yticks(y_ticks[::5])
plt.xticks(x[::5],x_ticks_labels[::5])
# todo:添加网格
plt.grid(True,linestyle='--',alpha=0.5)
# todo:添加描述对象
plt.ylabel('时间')
plt.xlabel('温度')
plt.title('一小时温度变化图')
# TODO:显示图例
plt.legend()
# TODO:将图show出来。
plt.show()