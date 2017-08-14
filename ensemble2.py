# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import copy
import cPickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import random
import scipy

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def process_data(raw_data, return_labels=True):

    # these are all the same across all employees, so remove them
    cols_to_remove = ['EmployeeCount', 'Over18', 'StandardHours', 
                      'DailyRate', 'HourlyRate', 'MonthlyRate', 
                      'PerformanceRating']
    for each_col in cols_to_remove:
        del raw_data[each_col]

    if return_labels:
        # pull out the label (attrition)
        labels = pd.DataFrame(raw_data['Attrition'])
        del raw_data['Attrition']
        labels = pd.Series(np.where(labels.Attrition.values == 'Yes', 1, 0),
                           labels.index)
    # Convert travel to corresponding ints    
    travel_dict = {"Non-Travel":0, "Travel_Rarely":1, "Travel_Frequently":2}
    raw_data.BusinessTravel.replace(travel_dict, inplace=True)


    # Add in some custom features

    raw_data['PotentialParent'] = ( scipy.stats.norm(25,5).pdf( raw_data['Age'] + 2*raw_data['Education'] ) 
                                    * pd.Series(np.where(raw_data.MaritalStatus.values == 'Married', 4, 1))
                                    * pd.Series(np.where(raw_data.Gender.values == 'Female', 3, 1)) )

    raw_data['IncomeProjection'] = raw_data['MonthlyIncome'] * ((1 + 0.01*raw_data['PercentSalaryHike'])**5)

    raw_data['RetirementProjection'] = raw_data['MonthlyIncome'] * ((1 + 0.01*raw_data['PercentSalaryHike'])**((65-raw_data['Age'])/5))

    raw_data['YearsPerJobLevel'] = raw_data['YearsAtCompany'] / raw_data['JobLevel']

    raw_data['DegTech'] = raw_data['EducationField'].map( {'Life Sciences':0, 'Medical':1, 'Technical Degree':2, 'Other':3, 'Marketing':4, 'Human Resources':5})
    raw_data['DegCrossField'] = (pd.Series(np.where(((raw_data.JobRole.values == 'Sales Representative') | (raw_data.JobRole.values == 'Sales Executive'))
                                                   & (raw_data.DegTech.values != 4),   1, 0))
                               + pd.Series(np.where(((raw_data.JobRole.values == 'Research Scientist') | (raw_data.JobRole.values == 'Research Director'))
                                                   & (raw_data.DegTech.values >= 4),   1, 0))
                               + pd.Series(np.where((raw_data.JobRole.values == 'Laboratory Technician') & (raw_data.DegTech.values >= 4), 1, 0))
                               + pd.Series(np.where((raw_data.JobRole.values == 'Healthcare Representative') & (raw_data.DegTech.values >=2), 1, 0))
                               + pd.Series(np.where((raw_data.JobRole.values == 'Human Resources') & (raw_data.DegTech.values != 5), 1, 0)))

    raw_data['WeightedDegCrossField'] = raw_data['DegCrossField'] * (raw_data['Education'])
    
    raw_data['CommuteSatisfaction'] = raw_data['DistanceFromHome'] / raw_data['EnvironmentSatisfaction']

    raw_data['UsedToCommute'] = raw_data['DistanceFromHome'] / (1.0 + raw_data['YearsAtCompany'])

    raw_data['YearsInCurrentSincePromotion'] = raw_data['YearsInCurrentRole'] - raw_data['YearsSinceLastPromotion']

    raw_data['PercentTimeWithCurrentManager'] = raw_data['YearsWithCurrManager'] / (1.0 + raw_data['YearsAtCompany']  )
 

    income_map_mean = {1:2800,2:5500,3:9800,4:15500,5:19000}
    raw_data['temp_col_mean'] = raw_data['JobLevel'].map( income_map_mean )
    raw_data['MonthlyMean'] = raw_data['MonthlyIncome'] - raw_data['temp_col_mean']
    raw_data.drop( 'temp_col_mean', axis=1, inplace=True )

    twy_map = {1:5.8,2:10.6,3:15.3,4:25.5,5:26.8}
    raw_data['temp_col'] = raw_data['JobLevel'].map( twy_map )
    raw_data['TWYMinusMean'] = raw_data['TotalWorkingYears'] - raw_data['temp_col']
    raw_data.drop( 'temp_col', axis=1, inplace=True )

  # these are categorical categories, we're going to change them to binary
    cols_to_transform_w_drop_first = ['Gender', 'MaritalStatus', 'OverTime']
    data = pd.get_dummies(raw_data, columns=cols_to_transform_w_drop_first, drop_first=True)

    cols_to_transform_no_drop = ['Department', 'EducationField', 'JobRole']
    data = pd.get_dummies(data, columns=cols_to_transform_no_drop, drop_first=False)

    # normalize each column
    data = (data - data.mean()) / data.std()

    if return_labels:
        return data, labels
    else:
        return data

def clean_data(data, cols):
    """
    Clean up the text by removing spaces and & from job titles and roles
    """
    for c in cols:
        data[c] = data[c].str.replace(' ', '')
        data[c] = data[c].str.replace('&', '')
        data[c] = data[c].str.replace('-', '_')
    return data


def bootstrap(train_data, test_data, num_features=5):
    """
    Randomly remove some features
    """
    cols_to_remove = np.random.choice(train_data.columns, 5, replace=False)
    bs_train = copy.deepcopy(train_data)
    bs_test = copy.deepcopy(test_data)
    for col in cols_to_remove:
        del bs_train[col]
        del bs_test[col]
    return bs_train, bs_test 


def main():
    raw_data = pd.read_csv('../train.csv')
    data, labels = process_data(raw_data)
    
    raw_test = pd.read_csv('../test.csv')
    test_data = process_data(raw_test, return_labels=False)

    # random forest classifier
    RF_clf = RandomForestClassifier(n_estimators=250,
				                    max_depth=10,
                				    max_features=10)

    # Decision tree classifier
    DT_clf = DecisionTreeClassifier(max_depth=None,
				    min_samples_split=2)

    # extra random forest classifier (best)
    ET_clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None,
				  min_samples_split=2)

    # naive bayes classifier
    NB_clf = GaussianNB()

    # SVM classifier
    class_weights = {0:1., 1:2.}
    SVC_clf = SVC(kernel='linear', probability=True, C=0.25,
                  class_weight=class_weights)
#    clf = BaggingClassifier(base_estimator=clfi, n_estimators=50, 
#                            max_samples=0.8, max_features=0.9, 
#                            bootstrap=True,
#                            bootstrap_features=True)

    #Nearest neighbor
    KNN_clf = KNeighborsClassifier(n_neighbors=12, weights='distance', p=3)

    # Gaussian process
    GP_clf = GaussianProcessClassifier(1.0*RBF(0.5), warm_start=True)

    # AdaBoost
    AB_clf = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
#    clf = BaggingClassifier(base_estimator=clfi, n_estimators=50, 
#                            max_samples=0.8, max_features=0.9, bootstrap=True,
#                            bootstrap_features=True)

    # MLP
    MLP_clf = MLPClassifier(hidden_layer_sizes=(60,50,20), activation='logistic', max_iter=500)
    MLP_clf_bag = BaggingClassifier(base_estimator=MLP_clf, n_estimators=50, 
                                    max_samples=0.8, max_features=0.9,
                                    bootstrap=True,
                                    bootstrap_features=True)

    classifiers = [       RF_clf, SVC_clf, KNN_clf, MLP_clf_bag, AB_clf]
    classifier_weights = [10,     50,      10,      30,          50]   
 
    num_outer = 1 # change random seed each time
    num_inner = 10  # redo train/test split
    clf_thresh = []
    clf_probs = []
    clf_predictions = []

    for outer_loop in xrange(num_outer):
        print("Outer: {0}".format(outer_loop))
        np.random.seed(outer_loop*25)
        random.seed(outer_loop*5)
        for inner_loop in xrange(num_inner):
            print("Inner: {0}".format(inner_loop))
            for clf in classifiers:

                # randly memove some columns (bootstrapping)
#                cv_data, bs_test_data  = bootstrap(data, test_data)
                cv_data = data
                bs_test_data = test_data

                # split the data into training and testing sets
                x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(
                    cv_data, labels, test_size=0.2)

                clf.fit(x_train_cv, y_train_cv)
                p_cv = clf.predict_proba(x_test_cv)
                cv_score = clf.score(x_test_cv, y_test_cv)*100.0
    
                # Compute ROC curve and ROC area for each class
                fpr, tpr, threshes = roc_curve(y_test_cv, p_cv[:, 1])
                roc_auc = auc(fpr, tpr)
        
                areas = 0.5*(1.0+np.abs(fpr-tpr))
                idx = np.argmax(areas)
                thresh = threshes[idx]
    
                print("Thresh: {0}  Areas: {1} FPR: {2} TPR: {3}".format(thresh,
                                                                         areas[idx],
                                                                         fpr[idx], 
                                                                         tpr[idx]))
                if roc_auc > 0.5: 
                    print("Classifier: {0}".format(clf))
                    print("Prediction score = %0.3f" % cv_score)
                    y_pred_cv = p_cv[:,1]
                    y_pred_cv[y_pred_cv < thresh] = 0.
                    y_pred_cv[y_pred_cv >= thresh] = 1.0
                    f2, t2, _ = roc_curve(y_test_cv, y_pred_cv)
                    roc_auc2 = auc(f2, t2)
                
                    cm = confusion_matrix(y_test_cv.astype(int), 
                                          y_pred_cv.astype(int)).astype(np.float32)
                    cm = cm/cm.sum(axis=1)[:, np.newaxis] *100.
                    print cm
                
                    print 'ROC area = %0.4f' % roc_auc
                    print 'ROC area (single point) = %0.4f' % roc_auc2
        
                    # run the test data through
                    test_probs = clf.predict_proba(bs_test_data)
                    clf_probs.append(test_probs)
        
                    test_predictions = copy.deepcopy(test_probs)
                    test_predictions[test_predictions >= thresh] = 1.
                    test_predictions[test_predictions < thresh] = 0.
                    clf_predictions.append(test_predictions)
                    
                    clf_thresh.append(thresh)
    
    pkl.dump((clf_probs, clf_predictions, clf_thresh), 
             open('ensemble_predictions.pkl', 'wb'))

    # print the results for a range of yes values
    n_yes_list = [50, 55, 60, 65, 70, 75 ]
    res_file_base = '../newFeats_RF_SVC_KNN_BMLP_AB_20I_20PCV'
    for n_yes in n_yes_list:
        res = pd.read_csv('../results.csv')
        res = res.assign(Attrition=np.average(clf_probs, axis=0))
        sorted_prob = np.sort(res.Attrition.values)
        thresh = 1 - sorted_prob[n_yes-1]
        print("%d thresh = %0.4f" % (n_yes, 1-thresh))
        res['Attrition'] = pd.Series(np.where(res.Attrition.values > 1-thresh, 
                                              'No', 'Yes'),
                                     res.Attrition.index)
        res.to_csv(res_file_base + '_y' + str(n_yes) + '.csv', index=False)

    # Uncomment the following to take the average threshold and the median
    # actual probability and write those to a file
#    final_predictions = np.mean(clf_probs, 0)
#    final_thresh = np.mean(clf_thresh)
#    final_predictions[final_predictions >= final_thresh] = 1.
#    final_predictions[final_predictions < final_thresh] = 0.
#    res = res.assign(Attrition=final_predictions)
#
#    res['Attrition'] = pd.Series(np.where(res.Attrition.values == 1., 'Yes', 'No'),
#                       res.Attrition.index)
#    res.to_csv('../SVC_AB_MLP_KNN_Probs_Bootstrap.csv', index=False)
#    
#    
#    final_predictions = np.round(np.median(clf_predictions, 0))
#
#    res = res.assign(Attrition=final_predictions)
#    res['Attrition'] = pd.Series(np.where(res.Attrition.values == 1., 'Yes', 'No'),
#                       res.Attrition.index)
#    res.to_csv('../SVC_AB_MLP_KNN_Preds_Bootstrap.csv', index=False)
#

if __name__ == "__main__":
    main()
