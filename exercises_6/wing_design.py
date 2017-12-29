import bitstring as bits
from random import randint

def indiv():
    return bits.BitArray('uint:25={}'.format(randint(0, 33554431)))#2^25-1

def fitness(bitstring):
    a = bitstring[0:6].uint
    b = bitstring[6:12].uint
    c = bitstring[12:18].uint
    d = bitstring[18:24].uint

    lift = (a-b)**2+(c+d)**2-(a-30)**3-(c-40)**3
    return lift

def tournament(indiv1, indiv2):
    result1 = fitness(indiv1)
    result2 = fitness(indiv2)

    if(result1 > result2):
        return indiv1
    return indiv2

def mutate(indiv, amount):
    new_indivs = []
    while len(new_indivs) < amount:
        new_indiv = indiv[:]
        for _ in range(1):
            indiv.invert(randint(0, 24))
        new_indivs.append(new_indiv)
    return new_indivs

def main():
    for _ in range(100):
        population = [indiv() for _ in range(100)]
        for _ in range(100):
            new_population = []
            for __ in range(25):
                indiv1 = population.pop(randint(0, len(population)-1))
                indiv2 = population.pop(randint(0, len(population)-1))
                new_population.append(tournament(indiv1, indiv2))
            for i in range(len(new_population)):
                new_population += mutate(new_population[i], 3)
            population = new_population[:]

        results = []
        for i in population:
            results.append((fitness(i), i))
        results.sort(key=lambda tup: tup[0])
        a = results[-1][1][0:6].uint
        b = results[-1][1][6:12].uint
        c = results[-1][1][12:18].uint
        d = results[-1][1][18:24].uint
        print(results[-1][0], ':', a, b, c, d)
    
    #print(results[0], results[-1])


        

if __name__ == '__main__':
    main()
