#!/usr/bin/python

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from ClassifyNB import classify

import matplotlib.pyplot as plt

features_train, labels_train, features_test, labels_test = makeTerrainData()

grade_fast = [features_train[ii][0] for ii in range(len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(len(features_train)) if labels_train[ii] == 1]

# Corrected classify call
clf, accuracy = classify(features_train, labels_train, features_test, labels_test)

# Visualize decision boundary
prettyPicture(clf, features_test, labels_test)
plt.savefig("test.png")

print("NB Accuracy:", accuracy)
