import numpy as np
from math import e
import math

def sigmoid(x):
    """Standard sigmoid; since it relies on ** to do computation, it broadcasts on vectors and matrices"""
    #return 1 / (1 - (e**(-x)))
    return 1 / (1 + math.exp(-x))
    
def derivative_sigmoid(x):
    """Expects input x to be already sigmoid-ed"""
    return x * (1 - x)

def tanh(x):
    """Standard tanh; since it relies on ** and * to do computation, it broadcasts on vectors and matrices"""
    return (1 - e ** (-2*x))/ (1 + e ** (-2*x))

def derived_tanh(x):
    """Expects input x to already be tanh-ed."""
    return 1 - tanh(x)


def forward(inputs,weights,function=sigmoid,step=-1):
    """
    Function needed to calculate activation on a particular layer.
    step=-1.0 calculates all layers, thus provides the output of the network
    step=0.0 returns the inputs
    any step in between, returns the output vector of that particular (hidden) layer
    """
    if step == 0:
        return inputs
    elif step == -1:
        step = len(weights) #go to output layer  
        
    output = np.append(1, inputs)
    for i in range(step):
        output = np.append(1, function(np.dot(weights[i], output))) #calculating activation
    return output[1:]
        
def backprop(inputs, outputs, weights, function=sigmoid, derivative=derivative_sigmoid, eta=0.01):
    """
    Function to calculate deltas matrix based on gradient descent / backpropagation algorithm.
    Deltas matrix represents the changes that are needed to be performed to the weights (per layer) to
    improve the performance of the neural net.
    :param inputs: (numpy) array representing the input vector.
    :param outputs:  (numpy) array representing the output vector.
    :param weights:  list of numpy arrays (matrices) that represent the weights per layer.
    :param function: activation function to be used, e.g. sigmoid or tanh
    :param derivative: derivative of activation function to be used.
    :param learnrate: rate of learning.
    :return: list of numpy arrays representing the delta per weight per layer.
    """
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    deltas = []
    layers = len(weights) # set current layer to output layer
    a_now = forward(inputs, weights, function, layers) # activation on current layer
    for i in range(0, layers):
        a_prev = forward(inputs, weights, function, layers-i-1) # calculate activation of previous layer
        if i == 0:
            error = np.array(derivative(a_now) * (outputs - a_now)).T  # calculate error on output
        else:
            error = np.expand_dims(derivative(a_now), axis=1) * weights[-i].T.dot(error)[1:] # calculate error on current layer
        delta = eta * np.expand_dims(np.append(1, a_prev), axis=1) * error.T # calculate adjustments to weights
        deltas.insert(0, delta.T) # store adjustments
        a_now = a_prev # move one layer backwards
    return deltas

# updating weights:
# given an array w of numpy arrays per layer and the deltas calculated by backprop, do
# for index in range(len(w)):
#     w[index] = w[index] + deltas[index]

def main():
    nor_inputs = np.array([([0, 0, 0], [1]),
                           ([1, 1, 1], [0]),
                           ([1, 0, 0], [0]),
                           ([0, 1, 0], [0]),
                           ([0, 0, 1], [0]),
                           ([1, 1, 0], [0]),
                           ([0, 1, 1], [0]),
                           ([1, 0, 1], [0])])
    rows = 1
    columns = 4
    x = np.array([0, 0, 0])
    theta = [np.random.rand(rows, columns)]

    print(theta)
    for _ in range(10000):
        for i in range(len(nor_inputs)):
            #print(nor_inputs[i][1])
            deltas = backprop(nor_inputs[i][0], nor_inputs[i][1], theta)
            theta = np.add(theta, deltas)

    for i in range(len(nor_inputs)):
        print(forward(nor_inputs[i][0], theta), ' -> ', nor_inputs[i][1])

if __name__ == '__main__':
    main()
