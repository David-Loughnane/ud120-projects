import sys
from time import time
#from pympler import tracker
from memory_profiler import memory_usage

import numpy as np
from sklearn import tree
from sklearn import metrics

from email_preprocess import preprocess



#memory_tracker = tracker.SummaryTracker()
print("Memory usage before: {}MB".format(memory_usage()))

### FEATURE EXTRACTION ####
features_train, features_test, labels_train, labels_test = preprocess()


print("features: ", features_train.shape)

#### INSTANTIATE CLASSIFIER ####
DTclassifier = tree.DecisionTreeClassifier(criterion = "entropy", min_samples_split=5, 
	max_features=20, max_depth=10, min_samples_leaf=3) 
# gini intended for continuous attributes - minimise misclassifcation
# entropy for data that occurs in classes - exploratroy analysis

#### TRAIN #####
t0 = time()
DTclassifier.fit(features_train, labels_train)
t1 = time()
print("Training time: {} seconds".format(round(t1-t0,3)))

#### PREDICT ####
t0 = time()
DTpreditiction = DTclassifier.predict(features_test)
t1 = time()
print("Prediction time: {} seconds".format(round(t1-t0,3)))
print("Prediciton accuracy: {:.2%}".format(metrics.accuracy_score(labels_test, DTpreditiction)))

print("Most important features: ", DTprediciton.feature_importances_)

#memory_tracker.print_diff()
print("Memory usage after: {}MB".format(memory_usage()))

