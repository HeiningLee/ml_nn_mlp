import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


def target_func(x1, x2):
    output = 2 * x1**2 + x2**2
    return output


def data_chart(network, testfile):
    """绘制测试图:生成测试集之后，显示一个散点图，同时显示测试数据和实际输出。"""
    # 建立图形
    fig = plt.figure().add_subplot(111, projection='3d')
    
    # 利用测试文件，先绘制测试数据图（红），再绘制输出数据图（蓝）
    with open(testfile, 'r') as fp:
        test_data_list = fp.readlines()  # 逐行读取
        for record in test_data_list:
            values = record.split(',')  # 行的分解
            inputs = numpy.asfarray(values[1:])/1000 + 0.5
            targets = float(values[0])/3000000
            outputs = network.query(inputs)

            # 逐点绘制散点图
            fig.scatter(inputs[0], inputs[1], targets, c='r')
            fig.scatter(inputs[0], inputs[1], outputs, c='b')
    plt.show()


def data_generation(dataname='data', sample_train=10000):
    # 将训练数据和测试数据一并生成，并没有太大的用处。
    # 生成训练数据和测试数据
    file_train = "train_" + dataname + '.csv'
    file_test = 'test_' + dataname + '.csv'
    sample_test = int(0.01 * sample_train)
    with open(file_train, 'w') as fp:
        for i in range(sample_train):
            a = 1000 * random.random() - 500  # 这里的变换是因为random使用的是(0,1)均匀分布
            b = 1000 * random.random() - 500
            y = target_func(a, b)
            fp.write(str(y))
            fp.write(',')
            fp.write(str(a))
            fp.write(',')
            fp.write(str(b))
            fp.write('\n')
            pass
        pass
    with open(file_test, 'w') as fp:
        for i in range(sample_test):
            a = 1000 * random.random() - 500
            b = 1000 * random.random() - 500
            y = target_func(a, b)
            fp.write(str(y))
            fp.write(',')
            fp.write(str(a))
            fp.write(',')
            fp.write(str(b))
            fp.write('\n')
            pass
        pass
    return file_train, file_test

