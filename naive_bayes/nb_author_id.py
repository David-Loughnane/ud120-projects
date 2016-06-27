import sys
from time import time
sys.path.append("../tools/")
#from pympler import tracker
from memory_profiler import memory_usage

from email_preprocess import preprocess

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

#memory_tracker = tracker.SummaryTracker()
print("Memory usage before: {}MB".format(memory_usage()))

features_train, features_test, labels_train, labels_test = preprocess()

print("features: ", features_train.shape)

NBclassifier = GaussianNB()

t0 = time()
NBclassifier.fit(features_train, labels_train)
t1 = time()
print("Training took: {} seconds".format(t1-t0))

NBpreditiction = NBclassifier.predict(features_test)
print(metrics.accuracy_score(labels_test, NBpreditiction))


#memory_tracker.print_diff()
print("Memory usage after: {}MB".format(memory_usage()))

