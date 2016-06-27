import numpy as np
import pylab as pl

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image


features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


NBclassifier = GaussianNB()
NBclassifier.fit(features_train, labels_train)

NBpreditiction = NBclassifier.predict(features_test)
print(metrics.accuracy_score(labels_test, NBpreditiction))
#print NBclassifier.score(features_test, labels_test)

### draw the decision boundary with the text points overlaid
prettyPicture(NBclassifier, features_test, labels_test, "naive_bayes/naive_bayes.png")
output_image("naive_bayes/naive_bayes.png", "png", open("naive_bayes/naive_bayes.png", "rb").read())

