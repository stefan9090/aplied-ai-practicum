import numpy as np
import math
import operator

def get_label(date, year):
    label = ""
    if date < year+301:
        return 'winter'
    elif year+301 <= date < year+601:
        return 'lente'
    elif year+601 <= date < year+901:
        return 'zomer'
    elif year+901 <= date < year+1201:
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

def get_shortest_distances(training_array, validation, labels, K):
    distances = []

    for i in range(len(training_array)):
        distances.append((calculate_distance(training_array[i], validation), labels[i]))

    distances.sort()
    
    shortest_distances = []
    for i in range(K):
        shortest_distances.append(distances[i])
    return shortest_distances

def get_prediction(training_array, validation, labels, K):
    seasons = {'herfst' : 0,
               'winter' : 0,
               'lente'  : 0,
               'zomer'  : 0}
    
    shortest_distances = get_shortest_distances(training_array, validation, labels, K)
    
    for i in shortest_distances:
        seasons[i[1]]+=1
    
    most_common_list = []
    most_found = 0
    for i in seasons:
        if seasons[i] > most_found:
            most_found = seasons[i]
            
    for i in seasons:
        if seasons[i] == most_found:
            most_common_list.append(i)
#    print(most_common_list)        
    if(len(most_common_list) > 1):
        for i in shortest_distances:
            if i[1] in seasons:
                return i[1]
    
    return most_common_list[0]
    
def main():
    k = 58
    training_labels = []
    validation1_labels = []

    training_data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0, 1, 2, 3, 4, 5 ,6 ,7])
    validation_data = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0, 1, 2, 3, 4, 5 ,6 ,7])

    dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
    dataset = []

    for label in dates:
        training_labels.append(get_label(label, 20000000))
    
    dates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
    for label in dates:
        validation1_labels.append(get_label(label, 20010000))        
    
    prediction_data = []
    for i in validation_data:
        prediction_data.append(get_prediction(training_data, i, training_labels, k))
   # print(prediction_data)
    failures = 0
    for i in range(len(validation1_labels)):
        if validation1_labels[i] != prediction_data[i]:
            failures += 1
    print(failures / len(validation1_labels) * 100)
            
    


    
if __name__ == '__main__':
    main()

    
