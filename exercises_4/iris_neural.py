import neural_network as nn
import numpy as np
import math
import random

setosa = 'Iris-setosa'
versicolor = 'Iris-versicolor'
virginica = 'Iris-virginica'

iris_names = {'Iris-setosa' : [0, 0, 1], 'Iris-versicolor' : [0, 1, 0], 'Iris-virginica' : [1, 0, 0]}

def tanh_der(x):
    return 1-(math.tanh(math.tanh(x)))

def relu(x):
    return max(0, x)

def relu_der(x):
    return x>0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

def main():

    training_size = 100
    validation_size = 50

    validation_data = []

    #load and open all dataset files
    training_values = np.genfromtxt('iris.data', delimiter=',', usecols=[0, 1, 2, 3])
    training_labels = np.genfromtxt('iris.data', delimiter=',', usecols=[4], dtype=str)
    training_data = []

    for i in range(len(training_values)):
        training_data.append([training_values[i], iris_names[training_labels[i]]])
    random.shuffle(training_data)

    
    for _ in range(validation_size):
        validation_data.append(training_data.pop())
    
    network = nn.Neural_network(4, [8, 3], sigmoid, sigmoid_der)
    for _ in range(400):
        for entry in training_data:
            network.backpropagate(entry, 0.1)

    good_counter = 0
            
    for entry in validation_data:
        result = [round(i, 0) for i in network.get_output(entry[0])]
        print(result, '->', entry[1])
        if result == entry[1]:
            good_counter+=1
    print(good_counter/validation_size * 100)
        #print([round(i, 0) for i in network.get_output(entry[0])], '->', entry[1])
    
        
if __name__ == '__main__':
    main()
