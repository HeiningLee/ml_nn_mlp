import numpy
from neuralnetwork import NeuralNetwork

nodes_input = 2
nodes_hidden = 100
output_nodes = 1

# learning rate
learning_rate = 0.1

# create instance of neural network
n = NeuralNetwork(nodes_input, nodes_hidden, output_nodes, learning_rate)
training_data_file = open("training_dataset.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# training the neural network

# epoches is the number of times the training data set is used for training.

epochs = 5

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        all_values = record.split(',')
        print(all_values)
        inputs = (numpy.asfarray(all_values[1:]) / 500.0) + 0.5
        targets = numpy.zeros(output_nodes) + 0.01

        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# load the mnist test data CSV file into a list
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])

    inputs = (numpy.asfarray(all_values) / 255.0* 0.99) + 0.01

    outputs = n.query(inputs)

    label = numpy.argmax(outputs)

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# calculate the performance score, the fraction of correct answers.
scorecard_array = numpy.asfarray(scorecard)
print("performance = ", scorecard_array.sum()) / scorecard_array.size
