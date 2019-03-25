import numpy


class NeuralNetwork:
    """神经网络类，基本的网络功能都在这里实现。
    注意激活函数是内嵌的sigmoid函数，想改变的话直接改动。
    """
    def __init__(self, nodes_in, nodes_hidden, nodes_out, learning_rate):
        # 设置节点数
        self.nodes_in = nodes_in
        self.nodes_hidden = nodes_hidden
        self.nodes_out = nodes_out

        # 设置学习率
        self.lr = learning_rate

        # 随机初始化权重矩阵
        self.wih = numpy.random.normal(0.0, pow(self.nodes_hidden, -0.5),
                                       (self.nodes_hidden, self.nodes_in))

        self.who = numpy.random.normal(0.0, pow(self.nodes_hidden, -0.5),
                                       (self.nodes_out, self.nodes_hidden))

        # 设置激活函数，这里使用的是sigmoid函数
        e = 2.7182818284590452353602874
        self.active_function = lambda x: 1/(1+pow(e, -1*x))

    def query(self, inputs_list):
        """网络查询函数，用于测试网络的前向通道。
        输入：网络输入；
        输出：网络输出。
        """
        # 将输入数据
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 隐藏层输入：对输入进行加权合并
        hidden_inputs = numpy.dot(self.wih, inputs)

        # 隐藏层输出：使用激活函数
        hidden_outputs = self.active_function(hidden_inputs)

        # 输出层的输入：对输入进行加权合并
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # 最终输出：使用激活函数
        final_outputs = self.active_function(final_inputs)

        return final_outputs

    def train(self, inputs_list, targets_list):
        # 将测试输入改为2维
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=1).T

        # 前向通道
        hidden_inputs = numpy.dot(self.wih, inputs)
        # print("inputs")
        # print(hidden_inputs)

        hidden_outputs = self.active_function(hidden_inputs)
        # print("outputs")
        # print(hidden_outputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.active_function(final_inputs)

        # 计算输出误差
        output_errors = targets - final_outputs

        # 误差反馈至隐藏层
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 调节隐藏层至输出层之间的权值矩阵
        self.who += self.lr * numpy.dot((output_errors * final_outputs *
                                        (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # 调节输入层至隐藏层之间的权值矩阵
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
                                        (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass
