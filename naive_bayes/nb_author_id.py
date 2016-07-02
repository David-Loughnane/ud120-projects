import sys
from time import time
sys.path.append("../tools/")
#from pympler import tracker
from memory_profiler import memory_usage

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

from email_preprocess import preprocess



#memory_tracker = tracker.SummaryTracker()
print("Memory usage before: {}MB".format(memory_usage()))

### FEATURE EXTRACTION ####
features_train, features_test, labels_train, labels_test = preprocess()
print(features_train)
print("features: ", features_train.shape)

#### INSTATNTIATE CLASSIFIER ####
NBclassifier = GaussianNB()

#### TRAIN #####
t0 = time()
NBclassifier.fit(features_train, labels_train)
t1 = time()
print("Training time: {} seconds".format(round(t1-t0,3)))

#### PREDICT ####
t0 = time()
NBpreditiction = NBclassifier.predict(features_test)
t1 = time()
print("Prediction time: {} seconds".format(round(t1-t0,3)))
print("Prediciton accuracy: {:.2%}".format(metrics.accuracy_score(labels_test, NBpreditiction)))

#memory_tracker.print_diff()
print("Memory usage after: {}MB".format(memory_usage()))