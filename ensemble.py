# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cPickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def clean_data(raw_data, return_labels=True):

    # these are all the same across all employees, so remove them
    cols_to_remove = ['EmployeeCount', 'Over18', 'StandardHours', 
                      'DailyRate', 'HourlyRate']
    for each_col in cols_to_remove:
        del raw_data[each_col]

    if return_labels:
        # pull out the label (attrition)
        labels = pd.DataFrame(raw_data['Attrition'])
        del raw_data['Attrition']
        labels = pd.Series(np.where(labels.Attrition.values == 'Yes', 1, 0),
                           labels.index)

    # these are categorical categories, we're going to change them to binary
    cols_to_transform = [ 'BusinessTravel', 'Department', 'EducationField',
                          'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

    data = pd.get_dummies(raw_data, columns=cols_to_transform, drop_first=True)

    # normalize each column
    data = (data - data.mean()) / (data.max() - data.min())

    if return_labels:
        return data, labels
    else:
        return data


def main():
    np.random.seed(5)
    raw_data = pd.read_csv('../train.csv')
    data, labels = clean_data(raw_data)

    # split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    # random forest classifier
    RF_clf = RandomForestClassifier(max_depth=None,
                                    random_state=0)

    # Decision tree classifier
    DT_clf = DecisionTreeClassifier(max_depth=None,
                                    random_state=0)

    # extra random forest classifier (best)
    ET_clf = ExtraTreesClassifier(max_depth=None,
                                  random_state=0)

    # naive bayes classifier
    NB_clf = GaussianNB()

    # SVM classifier
    SVC_clf = SVC(probability=True)
#    clf = BaggingClassifier(base_estimator=clfi, n_estimators=50, 
#                            max_samples=0.8, max_features=0.9, bootstrap=False,
#                            bootstrap_features=False)

    #Nearest neighbor
    KNN_clf = KNeighborsClassifier(weights='distance')

    # Gaussian process
    GP_clf = GaussianProcessClassifier(1.0*RBF(0.5))

    # AdaBoost
    AB_clf = AdaBoostClassifier()
#    clf = BaggingClassifier(base_estimator=clfi, n_estimators=50, 
#                            max_samples=0.5, max_features=0.6, bootstrap=True,
#                            bootstrap_features=True)


#    clf = VotingClassifier(estimators=[('RF', RF_clf), ('DT', DT_clf),
#                                       ('ET', ET_clf), ('NB', NB_clf),
#                                       ('SVC', SVC_clf), ('KNN', KNN_clf),
#                                       ('GP', GP_clf), ('AB', AB_clf)],
#                           voting='soft')
#
#    params = {'RF__n_estimators':[50, 2000], 'RF__min_samples_split':[1,5],
#              'DT__min_samples_split':[1,5], 'ET__n_estimators':[50, 2000],
#              'ET__min_samples_split':[1,5], 'KNN__n_neighbors': [1,25],
#              'KNN__p':[1,5], 'AB__n_estimators':[50,2000], 
#              'GP__warm_start':[False, True], 'AB__learning_rate':[0.01, 1.],
#              'SVC__kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
#    grid = GridSearchCV(estimator=clf, param_grid=params, cv=5)
#    grid.fit(x_train, y_train)
#    clf = VotingClassifier(estimators=[('RF', RF_clf), ('SVC', SVC_clf),
#                                       ('KNN', KNN_clf), ('AB', AB_clf)], 
#                           voting='soft')

#    params = {'RF__n_estimators':[50, 100, 250, 500, 1000, 2000], 
#              'RF__min_samples_split':[1,5],
#              'KNN__n_neighbors': [1,25],'KNN__p':[1,5], 
#              'AB__n_estimators':[50, 100, 250, 500, 1000, 2000],
#              'AB__learning_rate':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
#              'SVC__kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
#    grid = GridSearchCV(estimator=clf, param_grid=params, cv=5, n_jobs=8, verbose=5)
#
    grid = pkl.load(open('grid_search.pkl', 'rb'))
    #grid.fit(x_train, y_train)
    #pkl.dump(grid, open('grid_search.pkl', 'w'))
    p = grid.predict_proba(x_test)
    cv_score = grid.score(x_test, y_test)*100.0
    print("Prediction score = %0.3f" % cv_score)


    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshes = roc_curve(y_test, p[:, 1])
    roc_auc = auc(fpr, tpr)
    
    areas = 0.5*(1.0+np.abs(fpr-tpr))
    idx = np.argmax(areas)
    thresh = threshes[idx]

    print("Thresh: {0}  Areas: {1} FPR: {2} TPR: {3}".format(thresh, areas[idx], 
                                                            fpr[idx], tpr[idx]))

    y_pred = p[:,1]
    y_pred[y_pred < thresh] = 0.
    y_pred[y_pred >= thresh] = 1.0
    f2, t2, _ = roc_curve(y_test, y_pred)
    roc_auc2 = auc(f2, t2)

    cm = confusion_matrix(y_test.astype(int), y_pred.astype(int)).astype(np.float32)
    cm = cm/cm.sum(axis=1)[:, np.newaxis] *100.
    print cm

    print 'ROC area = %0.4f' % roc_auc
    print 'ROC area (single point) = %0.4f' % roc_auc2


    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # run the test data through

    raw_test = pd.read_csv('../test.csv')
    test_data = clean_data(raw_test, return_labels=False)
    clf.fit(data, labels)

    res = pd.read_csv('../results.csv')
#    res = res.assign(Attrition=clf.predict(test_data))
    res = res.assign(Attrition=clf.predict_proba(test_data))

    thresh_bump = 0.025
    res['Attrition'] = pd.Series(np.where(res.Attrition.values > 1-thresh + thresh_bump, 'No', 'Yes'),
                       res.Attrition.index)
    res.to_csv('../SKLearn_Results.csv', index=False)
    print('Threshold: {0}'.format(1-thresh+thresh_bump))


if __name__ == "__main__":
    main()
