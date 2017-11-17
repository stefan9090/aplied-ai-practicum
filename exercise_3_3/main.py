import numpy as np
import math

def calculate_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2 +
        (point1[2] - point2[2]) ** 2 +
        (point1[3] - point2[3]) ** 2 +
        (point1[4] - point2[4]) ** 2 +
        (point1[5] - point2[5]) ** 2 +
        (point1[6] - point2[6]) ** 2 
    )

def get_distance(training_array, validation):
    distances = []

    for i in training_array:
        distances.append(calculate_distance(i, validation))

    print(distances)
    
def main():
    dataset = []
    validation1 = []

    training_data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1, 2, 3, 4, 5 ,6 ,7])
    validation_data = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1, 2, 3, 4, 5 ,6 ,7])

    dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
    dataset = []
    for label in dates:
        if label < 20000301:
            dataset.append('winter')
        elif 20000301 <= label < 20000601:
            dataset.append('lente')
        elif 20000601 <= label < 20000901:
            dataset.append('zomer')
        elif 20000901 <= label < 20001201:
            dataset.append('herfst')
        else:
            dataset.append('winter')

    dates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
            
    for label in dates:
        if label < 20010301:
            validation1.append('winter')
        elif 20010301 <= label < 20010601:
            validation1.append('lente')
        elif 20010601 <= label < 20010901:
            validation1.append('zomer')
        elif 20010901 <= label < 20011201:
            validation1.append('herfst')
        else:
            validation1.append('winter')

    get_distance(training_data, validation_data[0])

    
if __name__ == '__main__':
    main()

    
