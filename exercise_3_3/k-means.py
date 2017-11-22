import numpy as np
import math
import operator
import random
import time


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
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2 +
        (point1[2] - point2[2]) ** 2 +
        (point1[3] - point2[3]) ** 2 +
        (point1[4] - point2[4]) ** 2 +
        (point1[5] - point2[5]) ** 2 +
        (point1[6] - point2[6]) ** 2 
    )

def calc_centr_grav(cluster, entry_count):
    data_set_len = len(cluster[0])

    data_set_avrg = [0] * data_set_len
    
    for data_set in cluster:
        for entry in range(data_set_len):
            data_set_avrg[entry] += data_set[entry]
    for i in range(data_set_len):
        data_set_avrg[i] /= entry_count
    return data_set_avrg

def gen_centroids(training_data, K):
    data_set_len = len(training_data[0])
    
    min_set = [0] * data_set_len #np.empty(data_set_len, dtype=int)
    max_set = [0] * data_set_len #np.empty(data_set_len, dtype=int)
    
    for data_set in training_data:
        for entry in range(data_set_len):
            if data_set[entry] < min_set[entry]:
                min_set[entry] = data_set[entry]
            elif data_set[entry] > max_set[entry]:
                max_set[entry] = data_set[entry]

    centroids = []
    for i in range(K):
        centroid = []
        for i in range(data_set_len):
            centroid.append(random.randrange(min_set[i], max_set[i])) 
        centroids.append(centroid)
    return centroids

def gen_clusters(training_data, centroids, K):
    clusters = []

    for i in range(K):
        clusters.append([])
    
    for data_set in training_data:
        closest_centroids = []
        for i in range(K):
            closest_centroids.append(calculate_distance(centroids[i], data_set))
        closest_index = closest_centroids.index(min(closest_centroids))
        clusters[closest_index].append(data_set)
        #print(str(centroids[closest_index]) + ' -> ' + str(data_set))
    return clusters
        
def k_means(training_file, validation_file, K):
    dates = np.genfromtxt(training_file, delimiter=';', usecols=[0])

    training_labels = []
    validation_labels = []
    
    for label in dates:
        training_labels.append(get_label(label, 20000000))
    
    for label in dates:
        validation_labels.append(get_label(label, 20010000))
    
    training_data = np.genfromtxt(training_file, delimiter=';', usecols=[1, 2, 3, 4, 5 ,6 ,7])
    validation_data = np.genfromtxt(validation_file, delimiter=';', usecols=[1, 2, 3, 4, 5 ,6 ,7])

    centroids = gen_centroids(training_data, K)
    
    clusters = gen_clusters(training_data, centroids, K)
    for i in range(K):
        entry_count = len(clusters[i])
        if entry_count > 0:
            centroids[i] = calc_centr_grav(clusters[i], entry_count)

    for i in clusters:
        print(len(i))
    print('-------------------')
    
    for _ in range(1000):
        clusters = gen_clusters(training_data, centroids, K)
        for i in range(K):
            entry_count = len(clusters[i])
            if entry_count > 0:
                centroids[i] = calc_centr_grav(clusters[i], entry_count)

    for i in clusters:
        print(len(i))
    print('-------------------')
        
def main():
    k_means('dataset1.csv', 'validation1.csv', 2)
    
if __name__ == '__main__':
    main()
