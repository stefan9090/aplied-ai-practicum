import functools as func
import decimal
import operator
import math
import random

class Input_neuron:
    def __init__(self, output=0):
        self.output = output

    def set_output(self, output):
        self.output = output

    def get_output(self):
        return self.output

    def get_output_sig(self):
        return self.output

    def get_output_sig_rec(self):
        return self.output

    
class Neuron:
    def __init__(self, inputs, bias=-1):
        self.inputs = []
        self.weights = []

        self.inputs.append(Input_neuron(bias))
        self.weights.append(-1)
        
        self.bias = bias
        self.bias_weight = random.uniform(-1.0, 1.0)
        
        self.output = 0
        
        for i in inputs:
            self.inputs.append(i[0])
            self.weights.append(i[1])

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
            
    def update(self, learning_rate, expected_output):
        output = self.get_output_sig()
        new_weights = []
        for i in range(len(self.inputs)):
            new_weights.append(self.weights[i] + learning_rate * self.inputs[i].output * self.sigmoid_der(self.get_output()) * (expected_output - output))
        self.weights = new_weights
           
    def get_output(self):
        output = 0
        for i in range(len(self.inputs)):
            output += self.inputs[i].output * self.weights[i]
        return output

    def get_output_rec(self):
        output = 0
        for i in range(len(self.inputs)):
            output += self.inputs[i].get_output() * self.weights[i]
        return int(output>0)
    
    def get_output_sig(self):
        buf = self.get_output()
        self.output = self.sigmoid(buf)
        return self.output

    def get_output_sig_rec(self):
        for i in self.inputs:
            i.get_output_sig_rec()
        buf = self.get_output()
        self.output = self.sigmoid(buf)
        return self.output
        
    
def NOR(input1, input2, input3):
    neuron = Neuron([(Input_neuron(input1), -1), (Input_neuron(input2), -1), (Input_neuron(input3), -1)], -1)
    return neuron.get_output()

class Neural_gate:
    def __init__(self):
        self.i1 = Input_neuron(0)
        self.i2 = Input_neuron(0)
        self.i3 = Input_neuron(0)

        self.neuron = Neuron([(self.i1, random.uniform(-1.0, 1.0)), (self.i2, random.uniform(-1.0, 1.0)), (self.i3, random.uniform(-1.0, 1.0))], 1)

    def train(self, input, learning_rate):    
        self.i1.set_output(input[0][0])
        self.i2.set_output(input[0][1])
        self.i3.set_output(input[0][2])
        
        outcome = self.neuron.get_output_sig_rec()
        self.neuron.update(learning_rate, input[1])
        
    def get_output(self, input):
        self.i1.set_output(input[0])
        self.i2.set_output(input[1])
        self.i3.set_output(input[2])
        return self.neuron.get_output_sig()

def adder(input1, input2):
    first_gate = Neuron([(Input_neuron(input1), -0.5), (Input_neuron(input2), -0.5)], -1)
    second_gate_top = Neuron([(Input_neuron(input1), -0.5), (first_gate, -0.5)], -1)
    second_gate_bot = Neuron([(first_gate, -0.5), (Input_neuron(input2), -0.5)], -1)
    
    outputS = Neuron([(second_gate_top, -0.5), (second_gate_bot, -0.5)], -1)
    outputC = Neuron([(first_gate, -0.5), (first_gate, -0.5)], -1)

    return outputS.get_output(), outputC.get_output()    

def main():
    nor_inputs = [([0, 0, 0], 1), ([1, 1, 1], 0), ([1, 0, 0], 0), ([0, 1, 0], 0), ([0, 0, 1], 0), ([1, 1, 0], 0), ([0, 1, 1], 0), ([1, 0, 1], 0)]
    nor = Neural_gate()

    for i in range(10000):
        for input in nor_inputs:
            nor.train(input, 0.5)

    print(nor.get_output([0, 0, 0]))
    
if __name__ == '__main__':
    main()
