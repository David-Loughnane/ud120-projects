import pickle
import sys
sys.path.append("../tools/")

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from feature_format import featureFormat, targetFeatureSplit


data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

predicitions = clf.predict(features_test)
accuracy = metrics.accuracy_score(labels_test, predicitions)
precision = metrics.precision_score(labels_test, predicitions) 
recall = metrics.recall_score(labels_test, predicitions)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)

conf_matrix = metrics.confusion_matrix(labels_test, predicitions)
print("Confusion matrix: ") 
print(conf_matrix)


target_names = ['blue collar', 'POI']
print(metrics.classification_report(labels_test, predicitions, target_names = target_names))
