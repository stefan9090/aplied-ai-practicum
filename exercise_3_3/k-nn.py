import numpy as np
import math
import operator
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
    
"""
returns a list of shortest distances, the length of the list is the given k. 
training_array = the main dataset(the data points we want the distance to)
validation = the point we want to predict the season of
k = how meany of the closes point do you want to return
"""
def get_shortest_distances(training_array, validation, labels, K):
    distances = []
    
    #creates a list of distances between every point in de data set and the given validation data point
    for i in range(len(training_array)):
        distances.append((calculate_distance(training_array[i], validation), labels[i]))
        
    distances.sort()
    
    #creates a list of distances from the sorted distances list that has the length of the given k 
    shortest_distances = []
    for i in range(K):
        shortest_distances.append(distances[i])
    return shortest_distances
    
"""
predicts what season a point is in.
training_array = the data set to base the prediction on
validation = the point to predict the seanon on
labels = the labels(seasons) that are assinged to the dataset.
k = the amount of data points to compair to. 
"""
def get_prediction(training_array, validation, labels, K):
    seasons = {'herfst' : 0,
               'winter' : 0,
               'lente'  : 0,
               'zomer'  : 0}
    
    shortest_distances = get_shortest_distances(training_array, validation, labels, K)
    
    #counts the amount of time a given season is in the list of shortest_distances
    for i in shortest_distances:
        seasons[i[1]]+=1
    
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
            
#   print(most_common_list)        
    if(len(most_common_list) > 1):
        for i in shortest_distances:
            if i[1] in seasons:
                return i[1]
    
    return most_common_list[0]
    
def main():
    training_labels = []
    validation1_labels = []
    
    #load and open all dataset files
    training_data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0, 1, 2, 3, 4, 5 ,6 ,7])
    validation_data = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0, 1, 2, 3, 4, 5 ,6 ,7])
    days_data = np.genfromtxt('days.csv', delimiter=';', usecols=[0, 1, 2, 3, 4, 5 ,6 ,7])

    #devide training data in seasons based on the date
    dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
    dataset = []

    for label in dates:
        training_labels.append(get_label(label, 20000000))
    
    #create list of labels from the dates of the validation data
    dates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
    for label in dates:
        validation1_labels.append(get_label(label, 20010000))        
    
    #predicts 100 time each time using k +1 and shows the error rate
    for k in range(100):
        prediction_data = []
        for i in validation_data:
            prediction_data.append(get_prediction(training_data, i, training_labels, k))
    # print(prediction_data)
        failures = 0
        for i in range(len(validation1_labels)):
            if validation1_labels[i] != prediction_data[i]:
                failures += 1
        
        print("k " + str(k) + " = " + str(failures / len(validation1_labels) * 100) + "% fout")
        
    #predicts the seasons of the days.cvs dataset
    days_prediction = []
    k = 58
    for f in days_data:
        days_prediction.append(get_prediction(training_data, f, training_labels, k))
    print(days_prediction)
    
    


    
if __name__ == '__main__':
    main()

    
