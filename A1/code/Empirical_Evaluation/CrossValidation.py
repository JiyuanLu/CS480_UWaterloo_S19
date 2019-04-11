import numpy as np
from utils.dataLoader import dataLoader
from models.DT import DT
from sklearn.model_selection import KFold
from models.KNN import KNN

def CrossValidation(N, classifier, k):
    '''
    N-fold Cross validation on the WDBC dataset, using classifier specified by clf.
    
    Inputs:
    - N: defines N-fold cross validation.
    - classifier: the classifier to use, DT (Decision Tree) or KNN (K Nearest Neighbors).
    - k: hyper-parameter for the classifier, i.e. k for KNN, maximum tree depth for DT.
    '''
    if classifier == 'DT':
        clf = DT()
    elif classifier == 'KNN':
        clf = KNN()
    loader = dataLoader()
    X_train, y_train, _, _ = loader.loadData()
    kf = KFold(n_splits=N, shuffle=True, random_state=0)
    average_precision = 0
    average_recall = 0
    average_accuracy = 0
    average_sensitivity = 0
    average_specificity = 0
    for train_index, val_index in kf.split(X_train):
        # N-fold cross validation
        X_tra, X_val = X_train[train_index], X_train[val_index]
        y_tra, y_val = y_train[train_index], y_train[val_index]
        
        clf.train(X_tra, y_tra, k)
        y_pred = clf.predict(X_val)
        num_val = y_val.shape[0]
        
        all_N = np.copy(y_val)
        all_N[:] = 'B'
        all_P = np.copy(y_val)
        all_P[:] = 'M'
        
        term_match = (y_pred == y_val)
        term_mismatch = np.logical_not(term_match)
        term_P = (y_pred == all_P)
        term_N = (y_pred == all_N)
        
        term_TP = np.logical_and(term_match, term_P)
        term_TN = np.logical_and(term_match, term_N)
        term_FP = np.logical_and(term_mismatch, term_P)
        term_FN = np.logical_and(term_mismatch, term_N)
        
        TP = np.sum(term_TP)
        TN = np.sum(term_TN)
        FP = np.sum(term_FP)
        FN = np.sum(term_FN)
        
        assert (TP + TN + FP + FN) == num_val
        
        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        accuracy = float(TP + TN) / (TP + TN + FP + FN)
        sensitivity = float(TP) / (TP + FN)
        specificity = float(TN) / (TN + FP)
        
        average_precision += precision
        average_recall += recall
        average_accuracy += accuracy
        average_sensitivity += sensitivity
        average_specificity += specificity
        
    average_precision /= N
    average_recall /= N
    average_accuracy /= N
    average_sensitivity /= N
    average_specificity /= N
    
    return average_precision, average_recall, average_accuracy, average_sensitivity, average_specificity
        
    
    
