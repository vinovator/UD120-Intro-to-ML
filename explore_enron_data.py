#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print(len(enron_data))  #46


# poi = 1 indicates that the particular person is "person of interest"
# count the no of poi in dataset
poi_count = len([person for person in enron_data if enron_data[person]["poi"]==1])
print(poi_count)  # 18

# compiled a list of all POI names (in ../final_project/poi_names.txt) 
# and associated email addresses (in ../final_project/poi_email_addresses.py)
# count how many poi in total
with open ("../final_project/poi_names.txt", "r") as poi:
    poi_text = poi.readlines()
    
# print(len(poi_text))  # 37, but includes empty space and reference text
# but all poi starts with either (n) or (y)
print(len([line for line in poi_text if line.startswith("(")]))

# Total stock value of James Prentise
print(enron_data["PRENTICE JAMES"]["total_stock_value"])

# How many email messages do we have from Wesley Colwell to persons of interest?
print(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

# What’s the value of stock options exercised by Jeffrey K Skilling?
print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

# Of these three individuals (Lay, Skilling and Fastow), 
# who took home the most money (largest value of “total_payments” feature)?
# How much money did that person get?
max_payment = max([enron_data[poi]["total_payments"] 
for poi in ("SKILLING JEFFREY K", "LAY KENNETH L", "FASTOW ANDREW S")])

poi = [poi for poi in ("SKILLING JEFFREY K", "LAY KENNETH L", "FASTOW ANDREW S") 
if enron_data[poi]["total_payments"]==max_payment]

print("{0}: {1}".format(poi, max_payment))
    

# How many folks in this dataset have a quantified salary? 
# What about a known email address?
print( len([poi for poi in enron_data if enron_data[poi]["salary"] != "NaN"])) # 95

print(len([poi for poi in enron_data if enron_data[poi]["email_address"] != "NaN"])) # 111

# what % of people in the dataset have "NaN" as total payments
print(len([poi for poi in enron_data 
     if enron_data[poi]["total_payments"] == "NaN"])*100/ len(enron_data)) # 14.38%

# How many POIs in the E+F dataset have “NaN” for their total payments? 
# What percentage of POI’s as a whole is this?
print(len([poi for poi in enron_data 
           if enron_data[poi]["poi"] == True 
           and enron_data[poi]["total_payments"]=="NaN"])) # 0
