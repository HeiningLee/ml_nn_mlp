"""这个例子来自《Python神经网络编程》（Tariq Rashid，林赐 译）。
本书介绍的神经网络实际上属于前馈神经网络中的MLP（多层感知机 Mulit-layered Perceptrons，
MLP）。

因为手头上没有 MNIST 数据集，并且对回归问题比较感兴趣，所以把例子改造成了一个对非线性函
数的回归问题。既然是回归问题，就不再使用准确率机制，而使用误差机制进行算法评价。

使用一个更高层次的循环，对世代数、隐节点数和学习率3个变量（简称为网络参数）决定的网络进行独立重复测试。
最后得到某一网络参数多次重复试验的方均根误差。

代码结构

main.py:            主函数；
neuralnetwork.py:   Neural Network 类的所在；
functions:          各类辅助方法，例如：拟合对象、生成数据、测试绘图等等。"""

import numpy
import math
from neuralnetwork import NeuralNetwork  # 用类实现一个标准的MLP
import functions as func

# 网络参数设置
nodes_input = 2
nodes_hidden = 50
nodes_output = 1

# 训练参数设置
file_train, file_test = func.data_generation('data', 10000)
learning_rate = 0.5  # 学习率
epochs = 20  # 世代数：使用同一个训练集反复训练同一个网络的重复次数

# 独立重复试验
ntests = 1
rmse = []
# 建立一个网络对象，在训练完成后，将最后一次训练的网络赋值给它，用于绘图
chart_net = NeuralNetwork(nodes_input, nodes_hidden, nodes_output, learning_rate)
"""创建神经网络"""
for test in range(0, ntests):
    print("\nTest:", test)
    # 创建神经网络实例
    n = NeuralNetwork(nodes_input, nodes_hidden, nodes_output, learning_rate)
    training_data_file = open(file_train, 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    """开始训练网络"""
    for e in range(epochs):
        print("Epochs:", e)
        # 遍历训练集中的每条数据
        for record in training_data_list:
            all_values = record.split(',')
            # 这里做了数据变换，目的是控制数值范围
            inputs = (numpy.asfarray(all_values[1:]) / 1000.0) + 0.5
            targets = float(all_values[0]) / 3000000
            n.train(inputs, targets)
            pass
        pass

    # 读取测试数据到一个列表中
    test_data_file = open(file_test, 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    chart_net = n

    """以下开始性能测试"""
    test_errors = []
    test_outputs = []

    # 待改：遍历测试集
    for record in test_data_list:
        all_values = record.split(',')
        # 测试集的标准答案
        targets = float(all_values[0]) / 3000000
        # 测试输入
        inputs = numpy.asfarray(all_values[1:]) / 1000 + 0.5
        # 测试输出
        outputs = n.query(inputs)
        test_outputs.append(outputs)  # 存储结果
        # 测试误差记录
        test_errors.append(targets - outputs)

    # 求解本次试验的RMSE
    mse = 0
    for error in test_errors:
        mse += error**2
    rmse.append(math.sqrt(mse/len(test_errors)))

    # SITREP
print("\n************* SITREP **************")
print("Nodes:", nodes_input, nodes_hidden, nodes_output)
print("LearningRate:", learning_rate)
print("Epochs:", epochs)
print("Tests:", ntests)
print("MRMSEs:", math.fsum(rmse)/len(rmse))

# 绘图
func.data_chart(chart_net, file_test)
