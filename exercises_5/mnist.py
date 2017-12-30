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
import random


def sigmoid(x):
    """Standard sigmoid; since it relies on ** to do computation, it broadcasts on vectors and matrices"""
    return 1 / (1 + (e**(-x))) #changed the '-' to a '+' because it didnt work otherwise
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
        step = len(weights) #go to output layer  
    output = np.append(1, inputs)
    for i in range(step):
        output = np.append(1, function(np.dot(weights[i], output))) #calculating activation
    return output[1:]

def backprop(inputs, outputs, weights, function=sigmoid, derivative=derivative_sigmoid, eta=0.01):
    """
    Function to calculate deltas matrix based on gradient descent / backpropagation algorithm.
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
            error = np.array(derivative(a_now) * (outputs - a_now))  # calculate error on output
        else:
            error = derivative(a_now) * (weights[-i].T).dot(error)[1:] # calculate error on current layer
        delta = eta * np.expand_dims(np.append(1, a_prev), axis=1) * error # calculate adjustments to weights
        deltas.insert(0, delta.T) # store adjustments
        a_now = a_prev # move one layer backwards

    return deltas

    
url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
if not os.path.isfile("mnist.pkl.gz"):
    urllib.request.urlretrieve(url, "mnist.pkl.gz")
f = gzip.open('mnist.pkl.gz', 'rb')
train_set , valid_set , test_set = pickle.load(f, encoding='latin1')
f.close()

def get_image(number, data_set=train_set):
    (X, y) = [img[number] for img in data_set]
    return (np.array(X), y)

def view_image(number, data_set=train_set):
    (X, y) = get_image(number, data_set)
    print("Label: %s" % y)
    plt.imshow(X.reshape(28,28), cmap=cm.gray)
    plt.show()
    
def main():
    image_count = len(train_set[0])

    nn_shape = [785] #the shape of the network including bias

    train_count = 10 #how oten should the network go through train_set

    buf = [] #generating random weights
    for i in nn_shape:
        buf.append([random.uniform(-1.0, 1.0) for _ in range(i)])
    network = np.array(buf)

    for count in range(train_count):
        for i in range(image_count):
            image, label = get_image(i)
            output = np.zeros(10)
            output[int(label)] = 1 #generate the expected output
            deltas = backprop(image, output, network) #get deltas
            network = np.add(network, deltas) #update weights with deltas
        print(str(count)+'/'+str(train_count))

    print('validation set:')
    good_counter = 0
    for i in range(len(valid_set[0])):
        image, label = get_image(i, valid_set)
        result = forward(image, network)
        if np.argmax(result) == int(label):
            good_counter+=1
        #print(np.argmax(result), '->', label)
    print(str(good_counter/len(valid_set[0])*100)+'%') #percentage of the time the network got the right answer

    print('test set:')
    good_counter = 0
    for i in range(len(test_set[0])):
        image, label = get_image(i, test_set)
        result = forward(image, network)
        if np.argmax(result) == int(label):
            good_counter+=1
        #print(np.argmax(result), '->', label)
    print(str(good_counter/len(valid_set[0])*100)+'%') #percentage of the time the network got the right answer

    
if __name__ == '__main__':
    main()
