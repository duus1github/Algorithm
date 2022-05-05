# 大部分的代码逻辑基本类似，就是说一些方法需要进行修改
import random
import matplotlib.pyplot as plt

# todo:初始化数据
# plt.rcParams['front.sans-self'] = ['SimHei']
plt.rcParams['font.sans-serif']=['SimHei']
x = range(60)
y_beijing = [random.randint(10, 20) for i in x]
y_wuyuan = [random.randint(20, 30) for i in x]
# todo:创建画图对象,返回两个对象，其中axes就是其中的子图的对象
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(20, 8))

# todo:绘制图例一和图例二
axes[0].plot(x, y_beijing, label='北京')
axes[1].plot(x, y_wuyuan, label='婺源', linestyle='--', color='b')
# todo:添加图例x,y刻度
x_label_ticks = ['11点{}分'.format(i) for i in x]
y_ticks = range(40)
axes[0].set_xticks(x[::5])
axes[0].set_yticks(y_ticks[::5])
axes[0].set_xticklabels(x_label_ticks[::5])
axes[1].set_xticks(x[::5])
axes[1].set_yticks(y_ticks[::5])
axes[1].set_xticklabels(x_label_ticks[::5])
# todo:添加网格
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[1].grid(True, linestyle='--', alpha=0.5)
# todo:添加title
axes[0].set_xlabel('时间')
axes[0].set_ylabel('温度')
axes[0].set_title('北京中午11点-12点温度变化', fontsize=25)
axes[1].set_xlabel('时间')
axes[1].set_ylabel('温度')
axes[1].set_title('婺源中午11点-12点温度变化', fontsize=25)
# todo:添加图例
axes[0].legend()
axes[1].legend()
plt.show()
