import sys
from time import time
from memory_profiler import memory_usage

import numpy as np
from sklearn import ensemble
from sklearn import tree
from sklearn import metrics

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
#plt.show()
################################################################################


print("Memory usage before: {}MB".format(memory_usage()))


print("features: ", len(features_train))

#### INSTANTIATE CLASSIFIER ####
ABclassifier = ensemble.AdaBoostClassifier(n_estimators=50, base_estimator=tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=100, max_depth=20))


#### TRAIN #####
t0 = time()
ABclassifier.fit(features_train, labels_train)
t1 = time()
print("Training time: {} seconds".format(round(t1-t0,3)))

print("The 1st classifier used in the ensemble method is:")
print(ABclassifier.estimators_[0])

#### PREDICT ####
t0 = time()
ABprediction = ABclassifier.predict(features_test)
t1 = time()
print("Prediction time: {} seconds".format(round(t1-t0,3)))
print("Prediciton accuracy: {:.2%}".format(metrics.accuracy_score(labels_test, ABprediction)))

print("Memory usage after: {}MB".format(memory_usage()))


################################################################################
try:
    prettyPicture(ABclassifier, features_test, labels_test)
except NameError:
    pass
