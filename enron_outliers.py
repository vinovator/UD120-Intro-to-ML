#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


def plot_data(data):
    ### your code below
    for point in data:
        salary = point[0]
        bonus = point[1]
        plt.scatter(salary, bonus)
        
    plt.xlabel("salary")
    plt.ylabel("bonus")
    plt.show()
    

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


# Plot the data
plot_data(data)

salary = list()
bonus = list()

for point in data:
    salary.append(point[0])
    bonus.append(point[1])
    
# find the array element with the highest bonus - outlier
max_item = [point for point in data if point[1]==max(bonus)]
print(max_item)
# this corresponds to TOTAL dictinonary key

# Find the key in data_dict dictionary where this max item belongs
for key, value in data_dict.items():
    if value["bonus"] == max_item[0][1]:
        print("Key corresponding to outlier is {0}".format(key))
        
# Pop the outlier out of the data set
print("Length of dict before removing outliers: {0}".format(len(data_dict)))
data_dict.pop("TOTAL")
print("Length of dict after removing outliers: {0}".format(len(data_dict)))

# Now call feature format again
data = featureFormat(data_dict, features)


# Plot the data again
plot_data(data)

# from the new plot it looks like atleast 2 people got 1 Mil+ salary and 5 Mil+ bonus
# find out who

for key, value in data_dict.items():
    if (float(value["bonus"]) > 5000000.0) and (float(value["salary"]) > 1000000.0):
        print (key)
        
# Unsurprusingly the names are LAY KENNETH L and SKILLING JEFFREY K

