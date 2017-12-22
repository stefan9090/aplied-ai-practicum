import functools as func
import decimal
import operator
import math
import random

class Input_neuron:
    def __init__(self, output=0):
        self.output = output
        self.weights = []
        self.inputs = []
        
    def set_output(self, output):
        self.output = output

    def get_output(self, g_func):
        return self.output

    def get_output_rec(self, g_func):
        return self.output

    def set_delta_hidden(self, error, outgoing_weight, g_der_func):
        return

    def update_backp(self, learnrate):
        return
    
class Neuron:
    def __init__(self, inputs, bias=1, min_random=-1.0, max_random=1.0):
        self.inputs = []
        self.weights = []
        
        self.inputs.append(Input_neuron(bias))
        self.weights.append(random.uniform(min_random, max_random))
        
        self.output = 0
        self.delta = 0

        for i in inputs:
            self.inputs.append(i[0])
            self.weights.append(i[1])

    def delta_func(self, i, learnrate, expected_output, output, g_der_func):
        return self.weights[i] + learnrate * self.inputs[i].output * g_der_func(self.get_node_input()) * (expected_output - output)

    def set_delta_output(self, expected_outcome, g_der_func):
        #sum_inputs = 0
        #for i in self.inputs:
        #    sum_inputs += i.output
        self.delta = g_der_func(self.get_node_input()) * (expected_outcome - self.output)
        
    def set_delta_hidden(self, next_layer, index, g_der_func):
        total_error = 0
        for neuron in next_layer:
            total_error += neuron.weights[index] * neuron.delta
        self.delta = g_der_func(self.get_node_input())*total_error

    def update(self, learning_rate, expected_output, g_func, g_der_func):
        output = self.get_output(g_func)
        new_weights = []
        for i in range(len(self.inputs)):
            new_weights.append(self.delta_func(i, learning_rate, expected_output, output, g_der_func))
        self.weights = new_weights

    def update_backp(self, learnrate):
        new_weights = []
        for i in range(len(self.inputs)):
            new_weights.append(self.weights[i] + learnrate * self.inputs[i].output * self.delta)
        self.weights = new_weights
        
    def get_node_input(self):
        output = 0
        for i in range(len(self.inputs)):
            output += self.inputs[i].output * self.weights[i]
        return output
    """
    def get_output_rec(self):
        output = 0
        for i in range(len(self.inputs)):
            output += self.inputs[i].get_node_input() * self.weights[i]
        return int(output>0)
    """    
    def get_output(self, g_func):
        buf = self.get_node_input()
        self.output = g_func(buf)
        return self.output

    def get_output_rec(self, g_func):
        for i in self.inputs:
            i.get_output_rec(g_func)
        buf = self.get_node_input()
        self.output = g_func(buf)
        return self.output
  
class Neural_network:
    def __init__(self, input_neuron_count, layer_structure, g_func, g_der_func, min_random=-1.0, max_random=1.0):
        self.network = []

        self.g_func = g_func
        self.g_der_func = g_der_func
        
        self.network.append([Input_neuron(0) for i in range(input_neuron_count)])
        network_index = 1

        for neuron_count in layer_structure:
            self.network.append([])
            for i in range(neuron_count):
                self.network[network_index].append(Neuron([(prev, random.uniform(min_random, max_random)) for prev in self.network[network_index-1]], 1, min_random, max_random))
            network_index+=1
       
    def feed_forward(self, inp, learning_rate):    
        for i in range(len(inp[0])):
            self.network[0][i].set_output(inp[0][i])
        
        outcome = self.network[-1][0].get_output_rec(self.g_func)
        self.network[-1][0].update(learning_rate, inp[1], self.g_func, self.g_der_func)

    def backpropagate(self, inp, learnrate):
        for i in range(len(inp[0])):
            self.network[0][i].set_output(inp[0][i])

        for output_neuron in self.network[-1]:
            output = output_neuron.get_output_rec(self.g_func)
            
        for layer in reversed(range(len(self.network))):
            for neuron in range(len(self.network[layer])):
                if self.network[layer][neuron] in self.network[-1]:
                    self.network[layer][neuron].set_delta_output(inp[1][neuron], self.g_der_func)
                else:
                    self.network[layer][neuron].set_delta_hidden(self.network[layer+1], neuron, self.g_der_func)
        for layer in self.network:
            for neuron in layer:
                neuron.update_backp(learnrate)
                
    def get_output(self, input):
        output_list = []
        for i in range(len(input)):
            self.network[0][i].set_output(input[i])
        for output_neuron in range(len(self.network[-1])):
            output_list.append(self.network[-1][output_neuron].get_output_rec(self.g_func))
        return output_list
            
    def __str__(self):
        string = ""
        for layer in range(len(self.network)):
            string += 'I:'
            for neuron in self.network[layer]:
                for i in range(len(neuron.inputs)):
                    string += str(round(neuron.inputs[i].output, 2)) + ' '
                string += '|'
            string += '\n'
            string += 'W:'
            for neuron in self.network[layer]:
                for w in range(len(neuron.weights)):
                    string += str(round(neuron.weights[w], 2)) + ' '
                string += '|'
            string += '\n'
        return string
