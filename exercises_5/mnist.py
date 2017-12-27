import os
import urllib.request
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.cm as cm
from math import e
import math


def sigmoid(x):
    """Standard sigmoid; since it relies on ** to do computation, it broadcasts on vectors and matrices"""
    return 1 / (1 - (e**(-x)))
    #return 1 / (1 + math.exp(-x))
    
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
    """Function needed to calculate activation on a particular layer.
    step=-1.0 calculates all layers, thus provides the output of the network
    step=0.0 returns the inputs
    any step in between, returns the output vector of that particular (hidden) layer"""
    if step == 0:
        return inputs
    elif step == -1:
        step = len(weights)
        
    output = np.append(1, inputs)
    for i in range(step):
        output = np.append(1, function(np.dot(weights[i], output)))
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


url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
if not os.path.isfile("mnist.pkl.gz"):
    urllib.request.urlretrieve(url, "mnist.pkl.gz")
f = gzip.open('mnist.pkl.gz', 'rb')
train_set , valid_set , test_set = pickle.load(f, encoding='latin1')
f.close()


def get_image(number):
    (X, y) = [img[number] for img in train_set]
    return (np.array(X), y)

def view_image(number):
    (X, y) = get_image(number)
    print("Label: %s" % y)
    plt.imshow(X.reshape(28,28), cmap=cm.gray)
    plt.show()

    
def main():
    image_count = 42000

    nn_shape = (3, 2, 1)
    network = np.ndarray(shape=nn_shape, dtype=float, order='F')
    print(network)
    """
    #theta = [np.random.rand(rows, columns+1)]
    for _ in range(1):
        for i in range(image_count):
            image, label = get_image(i)
            inp = np.zeros(10)
            inp[int(label)] = 1
            print(inp)
            deltas = backprop(get_image(i), inp, network)
            network = np.add(theta, deltas)
    """
    
    
if __name__ == '__main__':
    main()
