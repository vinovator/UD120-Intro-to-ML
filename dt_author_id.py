#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clf = DecisionTreeClassifier(min_samples_split=40)

print("Training begins")
t0 = time()
clf.fit(features_train, labels_train)
print("Training duration is {0}".format(round(time()-t0), "s"))
# for full data (3785 features) = 50 s
# with reduced feature set (379 features) = 5 s

print("Testing begins")
t1 = time()
pred = clf.predict(features_test)
print("Testing duration is {0}".format(round(time()-t1), "s"))
# for full data (3785 features) = 0 s
# with reduced feature set (379 features) = 0 s

score = accuracy_score(labels_test, pred)
# for full data (3785 features) = 97.78
# with reduced feature set (379 features) = 96.64
print("Accuracy is {0}".format(score))

#########################################################


