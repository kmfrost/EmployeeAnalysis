# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def clean_data(raw_data):

    # these are all the same across all employees, so remove them
    cols_to_remove = ['EmployeeCount', 'Over18', 'StandardHours']
    for each_col in cols_to_remove:
        del raw_data[each_col]

    # pull out the label (attrition)
    labels = pd.DataFrame(raw_data['Attrition'])
    del raw_data['Attrition']
    labels = pd.Series(np.where(labels.Attrition.values == 'Yes', 1, 0),
                       labels.index)

    # these are categorical categories, we're going to change them to binary
    cols_to_transform = [ 'BusinessTravel', 'Department', 'EducationField',
                          'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

    data = pd.get_dummies(raw_data, columns=cols_to_transform)

    # normalize each column
    data = (data - data.mean()) / (data.max() - data.min())

    return data, labels


def main():
    raw_data = pd.read_csv('../train.csv')
    data, labels = clean_data(raw_data)

    # split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    # random forest classifier
#    clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
#                                 min_samples_split=2, random_state=0)

    # Decision tree classifier
#    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
#                                 random_state=0)

    # extra random forest classifier (best)
    clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None,
                               min_samples_split=2, random_state=0)


    clf.fit(x_train, y_train)
    p = clf.predict_proba(x_test)

    fpr, tpr, _ = roc_curve(y_test, p[:,1])
    roc_auc = auc(fpr, tpr)
    print("ROC area = %0.2f", roc_auc)

if __name__ == "__main__":
    main()
