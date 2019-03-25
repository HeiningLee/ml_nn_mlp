import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成测试集之后，进行显示一个散点图
ax = plt.figure().add_subplot(111, projection='3d')

# 利用数据文件绘散点图
with open('data_training.csv', 'r') as fp:
    lines = fp.readlines()  # 逐行读取
    for line in lines:
        split_line = line.split(',')  # 行的分解
        data_line = []
        for i in split_line:
            data_line.append(float(i))  # 字符串转化为浮点数（只能一个一个转化）
        # 逐点绘制散点图
        ax.scatter(data_line[1], data_line[2], data_line[0], c='r')

plt.show()
