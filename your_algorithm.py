#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from time import time

# Gaussian training
print("Gaussian begins")
clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print("Training time {0}".format(round(time()-t0),"s")) # 0 s

t0 = time()
pred = clf.predict(features_test)
print("Testing time {0}".format(round(time()-t0),"s")) # 0 s

score = accuracy_score(labels_test, pred)
print("Accuracy score is {0}".format(score)) # 88.4


# SVM training
print("SVM begins")
clf = SVC(C=10000.0, kernel="rbf")

t0 = time()
clf.fit(features_train, labels_train)
print("Training time {0}".format(round(time()-t0),"s")) # 0 s

t0 = time()
pred = clf.predict(features_test)
print("Testing time {0}".format(round(time()-t0),"s")) # 0 s

score = accuracy_score(labels_test, pred)
print("Accuracy score is {0}".format(score)) # 93.2


# DTree training
print("Dtree begins")
clf = DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print("Training time {0}".format(round(time()-t0),"s")) # 0 s

t0 = time()
pred = clf.predict(features_test)
print("Testing time {0}".format(round(time()-t0),"s")) # 0 s

score = accuracy_score(labels_test, pred)
print("Accuracy score is {0}".format(score)) # 91.2


# KNeighbors training
print("K Nearest Neighbors begins")
# default n_neighbors = 5; weights = "uniform"
# n_neighbors should be less than no of samples
# clf = KNeighborsClassifier()

# default n_neighbors = 5; weights = "distance"
#clf = KNeighborsClassifier(weights="distance")

# n_neighbors = 10; weights = "distance"
clf = KNeighborsClassifier(n_neighbors = 10, weights="distance")

t0 = time()
clf.fit(features_train, labels_train)
print("Training time {0}".format(round(time()-t0),"s")) # 0 s

t0 = time()
pred = clf.predict(features_test)
print("Testing time {0}".format(round(time()-t0),"s")) # 0 s

score = accuracy_score(labels_test, pred)
print("Accuracy score is {0}".format(score))
# default n_neighbors = 5; weights = "uniform" - 92
# default n_neighbors = 5; weights = "distance" - 93.2


# Adaboost ensemble training
print("Adaboost ensemble begins")

# default n_estimators = 50
# clf = AdaBoostClassifier()

# n_estimators = 1000
clf = AdaBoostClassifier(n_estimators=1000)

t0 = time()
clf.fit(features_train, labels_train)
print("Training time {0}".format(round(time()-t0),"s")) # 0 s

t0 = time()
pred = clf.predict(features_test)
print("Testing time {0}".format(round(time()-t0),"s")) # 0 s

score = accuracy_score(labels_test, pred)
print("Accuracy score is {0}".format(score))
# default n_estimator = 50 - 92.4
# n_estimator = 1000 - 91.6


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
