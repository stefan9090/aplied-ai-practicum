import numpy as np
import math

def get_label(date, year):
    label = ""
    if date < year+301:
        return 'winter'
    elif 20000301 <= date < year+601:
        return 'lente'
    elif 20000601 <= date < year+901:
        return 'zomer'
    elif 20000901 <= date < year+1201:
        return 'herfst'
    else:
        return 'winter'


def calculate_distance(point1, point2):
    return math.sqrt(
        (point1[1] - point2[1]) ** 2 +
        (point1[2] - point2[2]) ** 2 +
        (point1[3] - point2[3]) ** 2 +
        (point1[4] - point2[4]) ** 2 +
        (point1[5] - point2[5]) ** 2 +
        (point1[6] - point2[6]) ** 2 +
        (point1[7] - point2[7]) ** 2 
    )

def get_shortest_distance(training_array, validation, dataset_labels, K):
    distances = []

    for i in range(len(training_array)):
        distances.append(calculate_distance(training_array[i], validation))

    distances.sort()
        
    print(distances[0])

    
def main():
    dataset_labels = []
    validation1_labels = []

    training_data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0, 1, 2, 3, 4, 5 ,6 ,7])
    validation_data = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0, 1, 2, 3, 4, 5 ,6 ,7])

    dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
    dataset = []

    for label in dates:
        dataset.append(get_label(label, 2000)
    
    dates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
            
    for label in dates:
        if label < 20010301:
            validation1_labels.append('winter')
        elif 20010301 <= label < 20010601:
            validation1_labels.append('lente')
        elif 20010601 <= label < 20010901:
            validation1_labels.append('zomer')
        elif 20010901 <= label < 20011201:
            validation1_labels.append('herfst')
        else:
            validation1_labels.append('winter')

    get_distance(training_data, validation_data[0])

    
if __name__ == '__main__':
    main()

    
