import numpy


class NeuralNetwork:
    #
    def __init__(self, nodes_in, nodes_hidden, nodes_out, learning_rate):
        # number of nodes
        self.nodes_in = nodes_in
        self.nodes_hidden = nodes_hidden
        self.nodes_out = nodes_out

        # learning rate
        self.lr = learning_rate

        # weight matrix
        self.wih = numpy.random.normal(0.0, pow(self.nodes_hidden, -0.5),
                                       (self.nodes_hidden, self.nodes_in))

        self.who = numpy.random.normal(0.0, pow(self.nodes_hidden, -0.5),
                                       (self.nodes_out, self.nodes_hidden))

        # weight matrix
        e = 2.7182818284590452353602874
        self.active_function = lambda x: 1/(1+pow(e, -1*x))

    def query(self, inputs_list):
        # convert input list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # signals emerging from hidden layer
        hidden_outputs = self.active_function(hidden_inputs)

        # calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # signals emerging from final output layer
        final_outputs = self.active_function(final_inputs)

        return final_outputs

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.active_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.active_function(final_inputs)

        # output_errors
        output_errors = targets - final_outputs
        # hidden errors
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the wih weights
        self.who += self.lr * numpy.dot((output_errors * final_outputs *
                                        (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # update the who weights
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
                                         (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass


