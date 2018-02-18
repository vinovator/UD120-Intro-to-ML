#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()

# Train the data
t0 = time()
clf.fit(features_train, labels_train)
print("Training time: {0}".format(round(time()-t0),"s"))

# testing the data
t1 = time()
pred = clf.predict(features_test)
print("Testing time: {0}".format(round(time()-t1),"s")) 

# Check the testing accuracy
test_score = accuracy_score(labels_test, pred)
print("Testing accuracy is {0}".format(test_score))

#########################################################


