import numpy as np
import pandas as pd
from src.utils.futils import *
import os
import yaml
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from math import comb, floor
from itertools import combinations, compress
from copy import copy
import rcca
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

"""
the structure of individualized yaml files (after adding param stats):

    - A1 (single valued, mean of allA1)
    - A1stdv (single valued, standard deviation of allA1)
    - A1range (single valued, max(allA1) - min(allA1))
    - A2 (single valued, mean of allA2)
    - A2stdv (single valued, standard deviation of allA2)
    - A2range (single valued, max(allA2) - min(allA2))
    - allA1 (list)
    - allA2 (list)
    - avg (single value, baseline glucose?)
    - lam (single value, mean of lam_list)
    - lamstdv (single valued, standard deviation of lam_list)
    - lamrange (single valued, max(lam_list) - min(lam_list))
    - lam_list (list)
    - stdv (single value)
    - umax (single value)
    - umax av (single value)
"""
"""
SCATTERPLOT AND CALCULATE CORRELATION OF THE FOLLOWING AGAINST HBA1C (label by color from diagnosis)
    - A1
    - A1stdv
    - A1range
    - A2
    - A2stdv
    - A2range
    - avg
    - lam
    - lamstdv
    - lamrange
    - stdv
    - umax
"""

def load_data_for_cca(params, params_to_include, include_diabetics=True):

    # dictionary of parameters (in terms of the dataframe) and variable names in the Parameters class
    attr = {'A1': 'A1', 'A1stdv': 'A1stdv', 'A1range': 'A1range', 'A2':'A2', 'A2stdv': 'A2stdv', 'A2range': 'A2range', 'avg':'avg', 'lam':'lam', 'lamstdv': 'lamstdv',
    'lamrange': 'lamrange', 'stdv':'stdv', 'umax':'umax', 'umax av': 'umax_av', 'Age': 'age', 'Sex': 'sex', 'Height (m)':'height', 'Weight (kg)': 'weight', 'BMI (kg/m^2)':'bmi',
    'Systolic Blood Pressure':'bp_sys', 'Diastolic Blood Pressure':'bp_dia', 'Resting HR (bpm)':'hr', 'Class':'status', 'HbA1c (%)':' a1c'}

    n_var = len(params_to_include)

    X = np.zeros([len(params.A1), n_var])
    keys = {params_to_include[i]: i for i in range(n_var)}

    for i, v in enumerate(params_to_include):
        attr_c = attr.get(v, 'none') # pulls the attribute name from dictionary
        if attr_c == 'none':
            continue
        temp = getattr(params, attr_c) # extracts attribute value from object

        X[:,i] = np.array(temp).reshape(-1)

    Y = np.array(params.a1c)

    if not include_diabetics: # remove the diabetics
        ind_to_remove = []
        for i in range(len(params.diags)):
            if params.diags[i] == 'Diabetic':
                ind_to_remove.append(i)
        X = np.delete(X, ind_to_remove, 0)
        Y = np.delete(Y, ind_to_remove, 0)
        

    return X, Y, keys

def augment_data(X, y):
    from scipy.optimize import curve_fit

    augmented_ind = [] # list to keep track of augmented params
    augmented_rate = []
    # normalization
    a1c_n = [x / max(y) for x in y]
    
    xs_min, xs_max = min(a1c_n), max(a1c_n)
    xs = np.linspace(xs_min,xs_max)
    ys_min, ys_max = min(X[:,1]), 1

    def model_func(x, a, k, b): # defines an exp functional prototype
        try:
            return a * np.exp(k*x) + b
        except:
            x = np.array(x)
            return a * np.exp(k*x) + b*np.ones(x.shape)

    for i in range(X.shape[1]): # fit each variable against target with exp and linear
        # linear fit
        poly = np.poly1d(np.polyfit(a1c_n, X[:,i], 1))

        # exp fit
        p0 = (1., 1.e-2, 2.)

        opt, _ = curve_fit(model_func, a1c_n, list(X[:,i]), p0, maxfev=5000)
        a, k, b = opt
        ys = model_func(xs, a, k, b)

        # computing SSE of fits
        sse_exp = np.mean(np.abs(X[:,i] - model_func(a1c_n, *opt)))
        sse_lin = np.mean(np.abs(X[:,i] - poly(a1c_n)))

        # sse_lin = 0.
        if np.abs((sse_exp - sse_lin) / sse_lin) < 0.01:
            # the same. do nothing
            continue
        elif np.abs((sse_exp - sse_lin) / sse_exp) < 0.01:
            # the same. do nothing
            continue
        elif sse_exp < sse_lin:
            # exp better. exp param stored in k, also keep current index
            X[:,i] = np.exp(k * X[:,i])
            augmented_ind.append(i)
            augmented_rate.append(k)
        else:
            # lin is better. do nothing
            continue

    return X, [augmented_ind, augmented_rate]

def processing(data, augment=True, getNormalization=False, data_params=[]):
    params_to_include = ['A1', 'A1stdv', 'A1range', 'A2', 'A2stdv', 'A2range', 'avg', 'lam', 'lamstdv', 'lamrange', 'stdv', 'Age', 'BMI (kg/m^2)']

    if getNormalization:
        # load data
        X, Y, _ = load_data_for_cca(data, params_to_include)

        # Normalizing each parameter by its max absolute value
        X_norm = np.max(np.abs(X), axis=0)
        Y_norm = np.max(Y)
        normalization = [X_norm, Y_norm]

        Xs = X / X_norm
        Ys = (Y / np.max(Y)).reshape(-1, 1)

        if augment:
            # augment data
            Xs, exp_params = augment_data(Xs, data.a1c)
            return Xs, Ys, normalization, exp_params
        else:
            return Xs, Ys, normalization, []

    else:
        # load data
        X, Y, _ = load_data_for_cca(data, params_to_include)

        # Normalizing each parameter by the normalization factors provided
        X_norm = data_params[0][0]
        Y_norm = data_params[0][1]
        
        Xs = X / X_norm
        Ys = (Y / Y_norm).reshape(-1, 1)

        if augment:
            ## Parameter augmentations
            exp_ind = data_params[1][0]
            exp_rate = data_params[1][1]

            for i in range(len(exp_ind)):
                Xs[:, exp_ind[i]] = np.exp(exp_rate[i] * Xs[:, exp_ind[i]])

        return Xs, Ys

def find_thresh(scores, data, eps=1e-6):
    # Placeholders for finding thresholds
    acc_best_upper = 0.
    acc_best_lower = 0.
    thresh_best_upper = 0.
    thresh_best_lower = 0.

    ## step 1: binary classification: separate diabetics from healthy/prediabetics. Positive = diabetic
    for i in range(scores.shape[0]): # set temp threshold as biomarker value + eps
        thresh_temp = scores[i] + eps
        pred = []
        TP = 0.
        FP = 0.
        TN = 0.
        FN = 0.
        # predict using the temp threshold
        for j in range(scores.shape[0]):
            if scores[j] < thresh_temp:
                pred.append('Not Diabetic')
            else:
                pred.append('Diabetic')
        
        for j in range(len(pred)):
            if pred[j] == 'Diabetic':
                if data.diags[j] == 'Diabetic':
                    TP += 1
                else:
                    FP += 1
            else:
                if data.diags[j] == 'Diabetic':
                    FN += 1
                else:
                    TN += 1

        # computes TPR (recall) and TNR (specificity)
        TPR = TP / (TP + FN)
        TNR = TN / (FP + TN)

        # computes balanced accuracy
        acc_balanced = (TPR + TNR) / 2
        acc = (TP + TN) / len(pred)

        if acc > acc_best_upper:
            acc_best_upper = acc
            thresh_best_upper = thresh_temp


    ## step 2: binary classification: do not look at diabetics. separate prediabetics from healthy. Positive = Prediabetic
    for i in range(scores.shape[0]):
        if scores[i] + eps > thresh_best_upper: # this is for finding the lower threshold. If lower threshold > upper threshold, then it's unfeasible and we skip
            continue
        else:
            thresh_temp = scores[i] + eps
            pred = []
            TP = 0.
            FP = 0.
            TN = 0.
            FN = 0.
            # predict using the temp threshold
            for j in range(scores.shape[0]):
                if scores[j] < thresh_temp:
                    pred.append('Non Diabetic')
                else:
                    pred.append('Not Healthy')
            for j in range(len(pred)):
                if data.diags[j] == 'Diabetic': # skips the diabetic cases
                    continue
                else:
                    if pred[j] == 'Non Diabetic': # predict healthy
                        if data.diags[j] == 'Non Diabetic':
                            TN += 1
                        else:
                            FN += 1
                    else: # predict prediabetic
                        if data.diags[j] == 'Non Diabetic':
                            FP += 1
                        else:
                            TP += 1
            
            # computes TPR (recall) and TNR (specificity)
            TPR = TP / (TP + FN)
            TNR = TN / (FP + TN)

            # computes balanced accuracy
            acc_balanced = (TPR + TNR) / 2
            acc = (TP + TN) / (TN + TP + FP + FN)

            if acc > acc_best_lower:
                acc_best_lower = acc
                thresh_best_lower = thresh_temp
                # print('new best lower threshhold. balaced acc: {}. acc: {}.'.format(acc_balanced, acc))

    print('Done Finding Thresholds!')

    return [thresh_best_lower, thresh_best_upper]

def train(data, augment=True):
    # data is an object from the Parameters() class
    # loads, normalizes, and augments data
    Xs, Ys, normalization, exp_params = processing(data, augment=augment, getNormalization=True, data_params=[])

    ## Initializes and tunes CCA
    print('Tuning GRM with CCA...')
    cca = rcca.CCA(kernelcca=False, reg=0., numCC=1)
    cca.train([Xs, Ys])

    weights_X = cca.ws[0]
    weights_Y = cca.ws[1]

    GRM = Xs.dot(weights_X.reshape(-1)) / weights_Y.reshape(-1)
    labels = Ys.reshape(-1)

    print('Tuning Done. GRM computed. Finding Thresholds')

    thresholds = find_thresh(GRM, data)

    return thresholds, cca.ws, [normalization, exp_params]
    
def predict(data, thresholds, weights, data_params, augment=True,):
    # loads, normalizes, and augments data
    Xs, Ys = processing(data, augment=augment, data_params=data_params)

    print('Predicting with GRM using given weights and thresholds...')
    # Compute GRM
    GRM = Xs.dot(weights[0].reshape(-1)) / weights[1].reshape(-1)
    labels = Ys.reshape(-1)

    thresh_best_lower = thresholds[0]
    thresh_best_upper = thresholds[1]
    # these two are just to build label and prediction as 1d arrays so they can be fed to generate ROC curve
    # 1 for healthy, 2 for prediabetic, 3 for diabetic
    final_pred = []
    final_label = []

    conf_mat = np.zeros([3,3])
    for i in range(GRM.shape[0]):
        if GRM[i] < thresh_best_lower: # predict healthy
            final_pred.append(1)
            if data.diags[i] == 'Non Diabetic': # healthy diag
                final_label.append(1)
                conf_mat[0,0] += 1
            elif data.diags[i] == 'Diabetic':
                final_label.append(3)
                conf_mat[2,0] += 1
            else:
                final_label.append(2)
                conf_mat[1,0] += 1
        elif GRM[i] > thresh_best_upper: # predict diabetic
            final_pred.append(3)
            if data.diags[i] == 'Non Diabetic':
                final_label.append(1)
                conf_mat[0,2] += 1
            elif data.diags[i] == 'Diabetic':
                final_label.append(3)
                conf_mat[2,2] += 1
            else:
                final_label.append(2)
                conf_mat[1,2] += 1
        else:
            final_pred.append(2)
            if data.diags[i] == 'Non Diabetic':
                final_label.append(1)
                conf_mat[0,1] += 1
            elif data.diags[i] == 'Diabetic':
                final_label.append(3)
                conf_mat[2,1] += 1
            else:
                final_label.append(2)
                conf_mat[1,1] += 1

    final_pred = np.array(final_pred)
    final_label = np.array(final_label)

    return GRM, final_pred, final_label, conf_mat
