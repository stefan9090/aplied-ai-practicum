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


def main():
    training_labels = []
    validation1_labels = []
    
    #load and open all dataset files
    values = np.genfromtxt('iris.data', delimiter=',', usecols=[0, 1, 2, 3])
    names = np.genfromtxt('iris.data', delimiter=',', usecols=[4], dtype=str)

    training_data = []
    
    for i in range(len(values)):
        training_data.append([values[i], iris_names[names[i]]])
    random.shuffle(training_data)
      
    network = nn.Neural_network(4, [5, 3], math.tanh, tanh_der)
    for _ in range(400):
        for entry in training_data:
            network.backpropagate(entry, 0.1)

    print(network.get_output(training_data[0][0]))
    
        
if __name__ == '__main__':
    main()
