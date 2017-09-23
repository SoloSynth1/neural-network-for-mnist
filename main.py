import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # three layers
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # learing rate
        self.lr = learning_rate

        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # activation function - sigmoid
        # scipy.special.expit()
        self.activation_function = lambda x: sp.expit(x)

    def train(self, inputs_list, targets_list):
        # same as query()
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        targets = np.array(targets_list, ndmin=2).T
        # error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# input = 28 x 28 pixel input = 784 nodes
inodes = 784
# hidden nodes = 100 (random)
hnodes = 100
# output = probability of 0 to 9 - 10 nodes
onodes = 10
# learning rate
learning_rate = 0.3

# read .csv
data = open('mnist_train.csv', 'r')
data_list = data.readlines()
data.close()

# initialize the neural network
n = NeuralNetwork(inodes, hnodes, onodes, learning_rate)

# train the network using the .csv data
for item in data_list:
    all_values = item.split(',')
    scaled_input = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    targets = np.zeros(onodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(scaled_input, targets)

# load test data
test_data = open('mnist_test.csv', 'r')
test_data_list = test_data.readlines()
test_data.close()

# choose a random MNIST data-set
query_input = test_data_list[np.random.randint(len(test_data_list))].split(',')
scaled_query_input = (np.asfarray(query_input[1:])/255.0 * 0.99) + 0.01

digit = int(query_input[0])
# show the digit
print('Target digit: %d'%(digit))
# show the output
result = n.query(scaled_query_input)
print result
print result[digit][0] / np.sum(result)

# image_array = np.asfarray(scaled_query_input).reshape((28, 28))
# plt.imshow(image_array, cmap='binary', interpolation='None')
# plt.show()
