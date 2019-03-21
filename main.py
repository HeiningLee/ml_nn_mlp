"""这个例子来自《Python神经网络编程》（Tariq Rashid，林赐 译）。
本书介绍的神经网络实际上属于前馈神经网络中的MLP（多层感知机 Mulit-layered Perceptrons，
MLP）。

因为手头上没有MNIST数据集，并且对回归问题比较感兴趣，所以把例子改造成了一个对非线性函
数的回归问题。既然是回归问题，就不再使用准确率机制，而使用误差机制进行算法评价。

使用一个更高层次的循环，对世代数、隐节点数和学习率3个变量（简称为网络参数）决定的网络进行独立重复测试。
最后得到某一网络参数多次重复试验的方均根误差。

代码结构
datagen.py:用于生成训练集和测试集；
main.py:主函数；
neuralnetwork.py:NeuralNetwork类的所在。

测试评论
"""


import numpy

from neuralnetwork import NeuralNetwork  # 用类实现一个标准的

nodes_input = 2
nodes_hidden = 100
output_nodes = 1

# 设置学习率，学习率影响权重调节的速率，我们希望它达到效率和稳定性的平衡。
learning_rate = 0.1

# 创建神经网络实例
n = NeuralNetwork(nodes_input, nodes_hidden, output_nodes, learning_rate)
training_data_file = open("training_dataset.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 开始网络训练

# 世代数设置。世代是指使用同一个训练集反复训练同一个网络的重复次数。

epochs = 5

for e in range(epochs):
    # 遍历训练集中的每条数据
    for record in training_data_list:
        all_values = record.split(',')
        print(all_values)
        inputs = (numpy.asfarray(all_values[1:]) / 500.0) + 0.5
        targets = numpy.zeros(output_nodes) + 0.01

        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# 读取测试数据到一个列表中
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

"""以下开始训练网络"""

# 待改：测试评分
scorecard = []

# 待改：遍历测试集
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])

    inputs = (numpy.asfarray(all_values) / 255.0 * 0.99) + 0.01

    outputs = n.query(inputs)

    label = numpy.argmax(outputs)

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# 待改：计算准确率评分。
scorecard_array = numpy.asfarray(scorecard)
print("performance = ", scorecard_array.sum()) / scorecard_array.size
