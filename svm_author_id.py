#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# For kernel = linear scenario

# clf = SVC(kernel="linear")
# for linear & original training set -  
# training time = 182 sec; testing time = 19 sec; accuracy = 98.4

# clf = SVC(kernel="rbf")
# for rbf & reduced training set -  
# training time = 0 sec; testing time = 1 sec; accuracy = 61.6

clf = SVC(C=10000.0, kernel="rbf")
# for rbf & reduced training set & playing with C  
# For C = 10.0 => accuracy = 61.6
# For C = 100.0 => accuracy = 61.6
# For C = 1000.0 => accuracy = 82.13
# For C = 10000.0 => accuracy = 89.24

# for rbf & full training set & C optimized to 10000.0,
# training time = 118 sec; testing time = 12 sec; accuracy = 99.08

# SVM is significantly slow with large dataset
# Slice the training set down to 1% of its original size
#features_train = features_train[:len(features_train)//100] 
#labels_train = labels_train[:len(labels_train)//100] 

# for linear & reduced training set -  
# training time = 0 sec; testing time = 1 sec; accuracy = 88.45

t0=time()
print("Fitting with SVM begins now")
clf.fit(features_train, labels_train)
print("Training time {0}".format(round(time()-t0),"s"))

t1=time()
print("Prediction with SVM begins now")
pred = clf.predict(features_test)
print("Testing time {0}".format(round(time()-t1),"s"))

test_score = accuracy_score(labels_test, pred)
print("Accuracy score for kernel SVM is {0}".format(test_score))

# predicting values for a particular data set using reduced dataset
# eventhough accuracy is a bit comprimised, trade off for speed
# Sarah - 0; Chris - 1

for row in (10, 26, 50):
    print("For row {0} the prediction is {author}".format(row, 
          author = "Sarah" if pred[row] == 0
          else "Chris"))  
# For row 10 the prediction is Chris
# For row 26 the prediction is Sarah
# For row 50 the prediction is Chris

pred_list = list(pred) # convert to list for easier syntax

# count the no of rows where the prediction is "Chris" = 1   
print("Length of test events: {0}; Chris: {1}; Sarah: {2}".format(len(pred_list),
      pred_list.count(1), pred_list.count(0)))

# Length of test events: 1758; Chris: 877; Sarah: 881

#########################################################


