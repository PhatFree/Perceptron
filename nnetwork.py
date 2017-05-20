import numpy
import random


#sigmoid func
def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+numpy.exp(-x))


class SigmoidNeuron:
    def __init__(self):
        self.input = 0
        self.output = 0

    def calculate_neuron(self, input):
        self.input = input
        self.output = sigmoid(input)
        return self.output

    def pass_through(self, input):
        self.input = input
        self.output = input
        return self.output


class NeuralNetwork:
    def __init__(self, input, hidden, output):
        self.input = input
        self.hidden = hidden
        self.output = output

        self.inputNeurons = None
        self.hiddenNeurons = None
        self.outputNeurons = None

        self.hidden_input = None
        self.hidden_output = None


def random_weights(r, c):
    return [[random.random()-.5 for _ in range(c)] for _ in range(r)]


def new_network(num_input, num_hidden, num_output):
    hidden_input = random_weights(num_input+1, num_hidden)
    hidden_output = random_weights(num_hidden+1, num_output)

    network = NeuralNetwork(num_input, num_hidden, num_output)
    network.inputNeurons = [SigmoidNeuron() for _ in range(num_input+1)]
    network.hiddenNeurons = [SigmoidNeuron() for _ in range(num_hidden+1)]
    network.outputNeurons = [SigmoidNeuron() for _ in range(num_output)]
    network.hidden_input = hidden_input
    network.hidden_output = hidden_output
    return network


# returns outputs [in list]
def calculate_network(network, inputs):
    if len(inputs) > network.input:
        return None

    # input layer
    for i in range(len(inputs)):
        network.inputNeurons[i].calculate_neuron(inputs[i])
    network.inputNeurons[-1].pass_through(1.0)  # and a bias neuron

    # hidden layer
    for h in range(network.hidden):
        sum = 0
        for i in range(network.input+1):
            sum += network.inputNeurons[i].output * network.hidden_input[i][h]
        network.hiddenNeurons[h].calculate_neuron(sum)
    network.hiddenNeurons[-1].pass_through(1.0)  # and a bias neuron

    # output layer
    for o in range(network.output):
        sum = 0
        for h in range(network.hidden+1):
            sum += network.hiddenNeurons[h].output * network.hidden_output[h][o]
        network.outputNeurons[o].calculate_neuron(sum)

    return [neuron.output for neuron in network.outputNeurons]


def back_propagate(network, data, expected, learn_rate):
    if len(data) > network.input:
        return None

    output = calculate_network(network, data)

    dO = [0.0]*network.output
    dH = [0.0]*network.hidden
    hidden_output_weights_prior = [[0.0]*(network.output+1)]*(network.hidden+1)

    for k in range(network.output):
        dO[k] = (expected[k] - output[k]) * output[k] * (1.0 - output[k])
        for j in range(network.hidden+1):
            hidden_output_weights_prior[j][k] = network.hidden_output[j][k]
            network.hidden_output[j][k] += learn_rate * dO[k] * network.hiddenNeurons[j].output

    for j in range(network.hidden):
        dH[j] = 0.0
        for k in range(network.output):
            dH[j] += dO[k] * hidden_output_weights_prior[j][k]
        dH[j] *= network.hiddenNeurons[j].output * (1.0 - network.hiddenNeurons[j].output)
        for i in range(network.input+1):
            network.hidden_input[i][j] += learn_rate * dH[j] * network.inputNeurons[i].output


def train_network(network, data, labels, learn_rate, epochs):

    for i in range(epochs):
        print('epoch:', i)
        for x in range(len(data)):
            back_propagate(network, data[x], labels[x], learn_rate)

