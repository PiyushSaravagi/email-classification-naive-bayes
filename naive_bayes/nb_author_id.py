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

########################################################
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

t0 = time()
classifier.fit(features_test, labels_test)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
predictions = classifier.predict(features_test)
print "testing time:", round(time()-t0, 3), "s"

correct_predictions = 0
total_predictions = len(predictions)

for index, p in enumerate(predictions):
    if p == labels_test[index]:
        correct_predictions += 1

accuracy = float(correct_predictions) / total_predictions
print "accuracy = " + str(accuracy)
#########################################################
