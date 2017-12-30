import random
import math
from functools import reduce
import operator


def individual(min_val, max_val):
    """
    This function returns an individual which consists of 10 unique numbers.
    """
    value_list = [i for i in range(min_val, max_val+1)] #generate a list of 1 to 10
    random.shuffle(value_list) #shuffle the list
    return value_list
    
def fitness(individual, divider, target_sum, target_multiply):
    """
    This function gets the fitness of an individual by calculating the average percentage off its intended target.
    """

    sum_val = reduce(operator.add, individual[:divider], 0)
    multiply_val = reduce(operator.mul, individual[divider:], 1)
    
    sum_error = abs(target_sum - sum_val)
    sum_error = sum_error / target_sum

    multiply_error = abs(target_multiply - multiply_val)
    multiply_error = multiply_error / target_multiply

    #print(multiply_error, sum_error)
    #print(sum_error, multiply_error)
    return (multiply_error + sum_error)/2 * 100

def tournament(indiv1, indiv2, divider):
    val1 = fitness(indiv1, divider, 36, 360)
    val2 = fitness(indiv2, divider, 36, 360)
    #print(val1, val2)
    if val1 > val2:
        return indiv2
    return indiv1

def mutate(indiv, divider, amount):
    """
    This function mutates an individual bij swapping 2 entries in both piles. It also check wether or not the new generated individual is an unique child
    """
    
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
            new_population = []
            for __ in range(25):
                indiv1 = population.pop(random.randint(0, len(population)-1)) #get 2 candidates voor the tournament
                indiv2 = population.pop(random.randint(0, len(population)-1))
                new_population.append(tournament(indiv1, indiv2, divider)) # append the winner to the new_population list
            for i in range(len(new_population)):
                new_population += mutate(new_population[i], divider, 3) #we mutate every winnen 3 times so we get a population of 100(including tournament winners) again. 
            population = new_population[:]

        #get best outcome based on the fitness function
        results = []
        for i in population:
            results.append((fitness(i, divider, 36, 360), i))
            #print(i)
        results.sort(key=lambda tup: tup[0])
        print(results[0])

def main():
    card_problem(5) #The variable is the divider.
"""
Wij verdelen de kaarten gelijk over beide stapels, omdat wij met deze verdeling precies kunnen uitkomen op 36 en 360. bij een andere verdeling lukte dit niet.
De afwijkingen die wij zien met een divider van 5 zijn tussen de 0 en de 4 procent, dit kan komen door lokaal optima.
"""
    
if __name__ == '__main__':
    main()
