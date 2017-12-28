import random
import math
from functools import reduce
import operator


def individual(min_val, max_val):
    value_list = [i for i in range(min_val, max_val+1)]
    random.shuffle(value_list)
    return value_list
"""
class card_individual:
    def __init__(self, min_val, max_val, values=None):
        if values == None:
            self.value_list = [i for i in range(min_val, max_val+1)]
            random.shuffle(self.value_list)
        else:
            self.value_list = values
            
        self.best_divider = 0

        self.sum_error = 0
        self.multiply_error = 0
"""        
def fitness(individual, divider, target_sum, target_multiply):
    sum_val = reduce(operator.add, individual[:divider], 0)
    multiply_val = reduce(operator.mul, individual[divider:], 1)
    
    sum_error = abs(target_sum - sum_val)
    sum_error = sum_error / target_sum

    multiply_error = abs(target_multiply - multiply_val)
    multiply_error = multiply_error / target_multiply

    #print(multiply_error, sum_error)
    #print(sum_error, multiply_error)
    return (multiply_error + sum_error)/2 * 100

def fitness2(individual, divider, target_sum, target_multiply):
    sum_val = 0
    multiply_val = 1
    
    for i in range(divider):
        sum_val += individual[i]

    for i in range(divider, len(individual)):
        multiply_val *= individual[i]

    multi_error = target_multiply - multiply_val
    sum_error = target_sum - sum_val
    
    error = math.sqrt(multi_error ** 2 + sum_error ** 2)
    return error


def tournament(indiv1, indiv2, divider):
    val1 = fitness(indiv1, divider, 36, 360)
    val2 = fitness(indiv2, divider, 36, 360)
    #print(val1, val2)
    if val1 > val2:
        return indiv2
    return indiv1

def mutate(indiv, divider, amount):
    new_indivs = []
    
    while len(new_indivs) < amount:
        index1 = random.randint(0, divider-1)
        index2 = random.randint(divider, len(indiv)-1)

        new_indiv = []
        for i in indiv:
            new_indiv.append(i)
            
        new_indiv[index1] = indiv[index2]
        new_indiv[index2] = indiv[index1]
        if new_indiv not in new_indivs:
            new_indivs.append(new_indiv)
    return new_indivs

def card_problem(divider):
    for x in range(100):
        population = [individual(1, 10) for _ in range(100)]
        for _ in range(100):
            results = []
            for __ in range(25):
                indiv1 = population.pop(random.randint(0, len(population)-1))
                indiv2 = population.pop(random.randint(0, len(population)-1))
                results.append(tournament(indiv1, indiv2, divider))
            new_population = []
            for i in results:
                new_population.append(i)
            for i in results:
                new_population += mutate(i, divider, 3)
            population = []
            for i in new_population:
                population.append(i)

        results = []
        for i in population:
            results.append((fitness(i, divider, 36, 360), i))
            #print(i)
        results.sort(key=lambda tup: tup[0])
        print(results[0])

def main():
    card_problem(5)
    
if __name__ == '__main__':
    main()
