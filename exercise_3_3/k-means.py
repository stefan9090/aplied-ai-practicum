import numpy as np
import math
import operator
import random
import time
import matplotlib.pyplot as plot

#determines a label based on the day of the year 
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

#calculate the distance between 2 points.
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
    
#returns the avrg distance whit in the data set to reposision the centroids
def calc_centr_grav(cluster, entry_count):
    data_set_len = len(cluster[0])

    data_set_avrg = [0] * data_set_len
    
    for data_set in cluster:
        for entry in range(data_set_len):
            data_set_avrg[entry] += data_set[entry]
            
    for i in range(data_set_len):
        data_set_avrg[i] /= entry_count
        
    return data_set_avrg
    
"""
generate k amount of centroids on random posisions 
the random range is set between the min and max distances of the datapoints 
"""
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
    
#assign the data points to the closses centroid 
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
    
#calculate the total length of all the data points to their centroid
def calculate_intra_cluster(clusters, centroids, K):
    total = 0
    for i in range(K):
        if len(clusters[i]) > 0:
            for entry in clusters[i]:
                total += calculate_distance(centroids[i], entry)
    return total
    
"""
generates k2-k10 multypol times and shows the result in a grafh
"""    
def plot_k(training_file, K):
    colors = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-']
    training_data = np.genfromtxt(training_file, delimiter=';', usecols=[0, 1, 2, 3, 4, 5 ,6 ,7])
    
    for color in colors:
        x = []
        y = []
    
        for K in range(2, 11):
            x.append(K)
            centroids = gen_centroids(training_data, K)
            for _ in range(10):
                clusters = gen_clusters(training_data, centroids, K)
                for i in range(K):
                    entry_count = len(clusters[i])
                    if entry_count > 0:
                        centroids[i] = calc_centr_grav(clusters[i], entry_count)

            y.append(calculate_intra_cluster(clusters, centroids, K))
        
        plot.plot(x, y, color)
    plot.show()   

"""
runs k_means whit a given k (best k we found was 3)
"""    
def k_means(training_file, K):
    dates = np.genfromtxt(training_file, delimiter=';', usecols=[0])
    
    training_labels = []
    validation_labels = []
    
    for label in dates:
        training_labels.append(get_label(label, 20000000))
        
    training_data = np.genfromtxt(training_file, delimiter=';', usecols=[0, 1, 2, 3, 4, 5 ,6 ,7])
    
    centroids = gen_centroids(training_data, K)
    clusters = gen_clusters(training_data, centroids, K)
    
    return clusters
    
def main():
    
    proper_clusters = False
    
    #keeps recreating clusters/centroids untill every cluster has somewhat of a sensable size
    while not proper_clusters:
        clusters = k_means('dataset1.csv', 3)
        proper_clusters = True
        
        for i in clusters:
            if len(i) < 25:
                #print("reclustering")
                proper_clusters = False
                
    print('\n' + "size of clusters")
    
    for i in clusters:
        print(len(i))
        
    print('\n' + "most common season in each cluster")
    for i in clusters:
        seasons = {'herfst' : 0,
                   'winter' : 0,
                   'lente'  : 0,
                   'zomer'  : 0}
                   
        for j in range(len(i)):
            seasons[get_label(i[j][0], 20000000)] += 1
        most_common_list = []
        most_found = 0
    
        #determins what season was found most often
        for i in seasons:
            if seasons[i] > most_found:
                most_found = seasons[i]
        #check to see if there are seasons that have been found the same amount of times as the most_found
        for i in seasons:
            if seasons[i] == most_found:
                most_common_list.append(i)

        print(most_common_list[0])
   
    print("plotting k k2 - k10")
    plot_k('dataset1.csv', 3)   
   
if __name__ == '__main__':
    main()
