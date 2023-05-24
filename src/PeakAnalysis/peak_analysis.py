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
### creates a class to store read parameter values (so it acts more like a struct). one entry per participant
# class Parameters():
#     def __init__(self):
#         ############## read directly from file
#         # model parameters from yaml
#         self.A1 = []
#         self.A1stdv = []
#         self.A1range = []
#         self.A2 = []
#         self.A2stdv = []
#         self.A2range = []
#         self.allA1 = []
#         self.allA2 = []
#         self.avg = []
#         self.lam = []
#         self.lamstdv = []
#         self.lamrange = []
#         self.lam_list = []
#         self.stdv = []
#         self.umax = []
#         self.umax_av = []

#         # standardized metrics
#         self.a1c = []
#         self.ogtt = []
#         self.fbg = []

#         # personal measurements
#         self.age = []
#         self.sex = []
#         self.height = []
#         self.weight = []
#         self.bmi = []
#         self.bp_sys = []
#         self.bp_dia = []
#         self.hr = []

#         # doctor's eval
#         self.diags = []
#         self.colors = [] # color based on dr's eval used for plotting purposes

#         # partcipant ID
#         self.participants = []

# def load_params_from_yaml(params, data, yaml_path, include_diabetics=True, specify_sex='all', min_peaks=25):
#     """
#     Inputs:
#         params - Parameters object to store all the values
#         participants - List of participants
#         yaml_path - path of yaml file to grab parameter values
#     Steps:
#         - opens yaml file specified in yaml_path
#         - check that each subject in the yaml file has a corresponding entry in the masterlist
#         - if its there, add everything to "params" object
#         - if its not there, go to the next case
#     Returns:
#         params - Parameters object with data filled
#         not_found - list of participants that have yaml files but not found in masterlist.csv
#     """
#     not_found = []
#     not_enough_peaks = []
#     ## Convert everything in the data frame to lists
#     # particpant list
#     participant_list = data['Participant ID'].to_list()

#     # personal measurements
#     age = data['Age'].to_list()
#     sex = data['Sex'].to_list()
#     height = data['Height (m)'].to_list()
#     weight = data['Weight (kg)'].to_list()
#     bmi = data['BMI (kg/m^2)'].to_list()

#     # standardized metrics
#     a1c = data['HbA1c (%)'].to_list()
#     ogtt = data['OGTT'].to_list()
#     fbg = data['FBG'].to_list()

#     # dr's eval
#     diags = data['Class'].to_list()

#     ### opens yaml file specified in yaml_path
#     # reads the list of yaml files
#     flist = []
#     for fname in os.listdir(yaml_path):
#         if fname[-5:] == '.yaml':
#             flist.append(fname)
    
#     ### check that each subject in the yaml file has a corresponding entry in the masterlist    
#     for id in flist: # iterates through participants
#         if id[:-5] not in participant_list: ## if the yaml is not on participant list
#             not_found.append(id[:-5])
#         else:
#             loc = participant_list.index(id[:-5]) # finds the index of the current subject in participant list

#             if (not include_diabetics) and (diags[loc] == 'Diabetic'): #skips the diabetic cases if needed
#                 continue
#             else:
#                 ### if its there, add everything to "params" object
#                 if specify_sex == 'all' or specify_sex == sex[loc]:
#                     # add params to object
#                     with open(yaml_path + id, 'r') as f:
#                         ps = yaml.safe_load(f)
#                     f.close()
#                     if len(ps['allA1']) < min_peaks:
#                         not_enough_peaks.append(id[:-5])
#                         print(id[:-5], len(ps['allA1']))
#                         continue
#                     else:
#                         params.A1.append(ps['A1'])
#                         params.A1stdv.append(ps['A1stdv'])
#                         params.A1range.append(ps['A1range'])
#                         params.A2.append(ps['A2'])
#                         params.A2stdv.append(ps['A2stdv'])
#                         params.A2range.append(ps['A2range'])
#                         params.allA1.append(ps['allA1'])
#                         params.allA2.append(ps['allA2'])
#                         params.avg.append(ps['avg'])
#                         params.lam.append(ps['lam'])
#                         params.lamstdv.append(ps['lamstdv'])
#                         params.lamrange.append(ps['lamrange'])
#                         params.lam_list.append(ps['lam_list'])
#                         params.stdv.append(ps['stdv'])
#                         params.umax.append(ps['umax'])
#                         params.umax_av.append(ps['umax av'])

#                         # add standardized metrics to object
#                         params.a1c.append(a1c[loc])
#                         params.ogtt.append(ogtt[loc])
#                         params.fbg.append(fbg[loc])
                        
#                         # add personal measurements to object
#                         params.age.append(age[loc])
#                         params.sex.append(sex[loc])
#                         params.height.append(height[loc])
#                         params.weight.append(weight[loc])
#                         params.bmi.append(bmi[loc])

#                         # add dr's eval, and the corresponding color to object
#                         params.diags.append(diags[loc])
#                         if diags[loc] == 'Diabetic':
#                             params.colors.append('r')
#                         elif diags[loc] == 'Prediabetic':
#                             params.colors.append('y')
#                         elif diags[loc] == 'Non Diabetic':
#                             params.colors.append('g')
#                         else:
#                             params.colors.append('')

#                         # add participant to object
#                         params.participants.append(id[:-5])

#                 ### if its not there, go to the next case   
#                 else:
#                     continue


#     ### return the params object
#     return params, not_found, not_enough_peaks

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
        p0 = (1., 1.e-5, 1.) # initial coefficient guess
        opt, _ = curve_fit(model_func, a1c_n, list(X[:,i]), p0, maxfev=5000)
        a, k, b = opt
        ys = model_func(xs, a, k, b)

        # computing SSE of fits
        sse_exp = np.mean(np.abs(X[:,i] - model_func(a1c_n, *opt)))
        sse_lin = np.mean(np.abs(X[:,i] - poly(a1c_n)))

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

def processing(data, getNormalization=False, data_params=[]):
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

        # augment data
        # Xs, exp_params2 = augment_data(Xs, data.a1c)

        # ## Parameter augmentation (NEED TO MAKE THIS NOT MANUAL)
        # Xs[:,0] = np.exp(-3.787044 * Xs[:,0]) # transform mu of A1 to exponential class
        # Xs[:,6] = np.exp(4.423684 * Xs[:,6]) # transform mu of ebar to exponential class
        # Xs[:,10] = np.exp(1.355863 * Xs[:,10]) # transform std dev of time series to exponential class
        
        exp_ind = [0, 6, 10]
        exp_rate = [-3.787044, 4.423684, 1.355863]
        exp_params = [exp_ind, exp_rate]

        print(exp_params)
        # print(exp_params2)

        # return Xs, Ys, normalization, exp_params2
        return Xs, Ys, normalization, exp_params
    else:
        # load data
        X, Y, _ = load_data_for_cca(data, params_to_include)

        # Normalizing each parameter by the normalization factors provided
        X_norm = data_params[0][0]
        Y_norm = data_params[0][1]
        
        Xs = X / X_norm
        Ys = (Y / Y_norm).reshape(-1, 1)

        ## Parameter augmentations
        exp_ind = data_params[1][0]
        exp_rate = data_params[1][1]
        # for i in range(len(exp_ind)):
        #     Xs[:, exp_ind[i]] = np.exp(exp_rate[i] * Xs[:, exp_ind[i]])

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
            # print('new best upper threshhold. balaced acc: {}. acc: {}.'.format(acc_balanced, acc))


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

def train(data):
    #   data is an object from the Parameters() class
    # loads, normalizes, and augments data
    Xs, Ys, normalization, exp_params = processing(data, getNormalization=True, data_params=[])

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
    
def predict(data, thresholds, weights, data_params):
    # loads, normalizes, and augments data
    Xs, Ys = processing(data, data_params=data_params)

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

# if __name__ == '__main__':

#     print(X_v2g.shape, Xs_v2g.shape)

#     GRM = Xs_v2g.dot(weights_X.reshape(-1)) / weights_Y.reshape(-1)
#     allY_v2g = Ys_v2g

#     # these two are just to build label and prediction as 1d arrays so they can be fed to generate ROC curve
#     # 1 for healthy, 2 for prediabetic, 3 for diabetic
#     final_pred = []
#     final_label = []

#     conf_mat_v2g = np.zeros([3,3])
#     for i in range(GRM.shape[0]):
#         if GRM[i] < thresh_best_lower: # predict healthy
#             final_pred.append(1)
#             if data.diags[i] == 'Non Diabetic': # healthy diag
#                 final_label.append(1)
#                 conf_mat_v2g[0,0] += 1
#             elif data.diags[i] == 'Diabetic':
#                 final_label.append(3)
#                 conf_mat_v2g[2,0] += 1
#             else:
#                 final_label.append(2)
#                 conf_mat_v2g[1,0] += 1
#         elif GRM[i] > thresh_best_upper: # predict diabetic
#             final_pred.append(3)
#             if data.diags[i] == 'Non Diabetic':
#                 final_label.append(1)
#                 conf_mat_v2g[0,2] += 1
#             elif data.diags[i] == 'Diabetic':
#                 final_label.append(3)
#                 conf_mat_v2g[2,2] += 1
#             else:
#                 final_label.append(2)
#                 conf_mat_v2g[1,2] += 1
#         else:
#             final_pred.append(2)
#             if data.diags[i] == 'Non Diabetic':
#                 final_label.append(1)
#                 conf_mat_v2g[0,1] += 1
#             elif data.diags[i] == 'Diabetic':
#                 final_label.append(3)
#                 conf_mat_v2g[2,1] += 1
#             else:
#                 final_label.append(2)
#                 conf_mat_v2g[1,1] += 1

#     final_pred = np.array(final_pred)
#     final_label = np.array(final_label)

#     #### OvR:
#     ## Note OvR does not really make sense because roc will alter the healthy/rest threshold and diabetic/rest threshold, but prediabetic/rest has two thresholds
#     # split into three sets for each OvR classification: healthy vs rest, prediabetic vs rest, diabetic vs rest
#     h_label = np.array([1 if i == 1 else 0 for i in final_label])
#     p_label = np.array([1 if i == 2 else 0 for i in final_label])
#     d_label = np.array([1 if i == 3 else 0 for i in final_label])

#     # flip the order so healthy is closer to 1
#     h_pred = GRM / np.max(GRM)
#     p_pred = copy(h_pred)
#     d_pred = copy(h_pred)

#     # flip the order so healthy is closer to 1
#     h_pred = np.ones(h_pred.shape) - h_pred

#     h_fpr, h_tpr, h_thresholds = roc_curve(h_label, h_pred)
#     p_fpr, p_tpr, p_thresholds = roc_curve(p_label, p_pred)
#     d_fpr, d_tpr, d_thresholds = roc_curve(d_label, d_pred)

#     h_prec, h_rec, _ = precision_recall_curve(h_label, h_pred)
#     p_prec, p_rec, _ = precision_recall_curve(p_label, p_pred)
#     d_prec, d_rec, _ = precision_recall_curve(d_label, d_pred)

#     h_pr = np.stack((h_fpr, h_tpr), axis=1)
#     h_pr[:,1] -= np.ones(h_pr[:,1].shape) # calculates displacement to (0,1) where TPR = 1, FPR = 0, i.e. the ideal scenario
#     d_pr = np.stack((d_fpr, d_tpr), axis=1)
#     d_pr[:,1] -= np.ones(d_pr[:,1].shape) # calculates displacement to (0,1) where TPR = 1, FPR = 0, i.e. the ideal scenario
    
#     hp_dists = np.linalg.norm(h_pr, axis=1) # calculates distance to (0,1)
#     hp_ind = np.argmin(hp_dists)

#     pd_dists = np.linalg.norm(d_pr, axis=1) # calculates distance to (0,1)
#     pd_ind = np.argmin(pd_dists)

#     print('for h/p: shortest distance is {} at index {}, with threshold {}, and renormalized {}'.format(min(hp_dists), hp_ind, h_thresholds[hp_ind], (1 - h_thresholds[hp_ind])*np.max(GRM)))
#     print('for p/d: shortest distance is {} at index {}, with threshold {}, and renormalized {}'.format(min(pd_dists), pd_ind, d_thresholds[pd_ind], d_thresholds[pd_ind]*np.max(GRM)))

    
#     print('healthy roc auc: {}'.format(roc_auc_score(h_label, h_pred)))
#     print('prediabetic roc auc: {}'.format(roc_auc_score(p_label, p_pred)))
#     print('diabetic roc auc: {}'.format(roc_auc_score(d_label, d_pred)))


#     ## AUC: 0.9143399810064477 for healthy
#     ## AUC: 0.9497354497354498 for diabetic

#     ax[1].plot(h_fpr, h_tpr, label='Non-Diabetic. AUC: {:.4f}'.format(roc_auc_score(h_label, h_pred)))
#     # ax[1].plot(p_fpr, p_tpr, label='Prediabetic. AUC: {:.4f}'.format(roc_auc_score(p_label, p_pred)))
#     ax[1].plot(d_fpr, d_tpr, label='Diabetic. AUC: {:.4f}'.format(roc_auc_score(d_label, d_pred)))
#     ax[1].plot([0,1], [0,1], '--')
#     ax[1].set_xlabel('False Positive Rate')
#     ax[1].set_ylabel('True Positive Rate')
#     ax[1].set_title('ROC for Klick-600')
#     ax[1].grid(alpha=0.4)
#     ax[1].legend(loc='lower right')

#     plt.show()

#     #### OvO:
#     ## Note: OvO makes more sense as long as wel only consider the "adjacent" classes, i.e. health/prediabetic and prediabetic/diabetic
#     ## roc only compares tpr and fpr which and so negative predictions are entirely ignored


#     # conf_mat = get_conf_mat(test='HbA1c')
#     prec, rec, f1, accs = get_scores_from_conf_mat(conf_mat_v2g)

#     Precision_H = prec[0]
#     Precision_P = prec[1]
#     Precision_D = prec[2]

#     Recall_H = rec[0]
#     Recall_P = rec[1]
#     Recall_D = rec[2]

#     F1_H = f1[0]
#     F1_P = f1[1]
#     F1_D = f1[2]
    
#     acc = accs[0]
#     bal_acc = accs[1]


#     print('-------------Klick600-------------')
#     print('Total Predictions: {}'.format(int(np.sum(conf_mat_v2g[:]))))
#     print('Males: {}'.format(np.sum(np.array([i == 'M' for i in data.sex]))))
#     print('Females: {}'.format(np.sum(np.array([i == 'F' for i in data.sex]))))
#     print('=======Healthy======')
#     print('Precision: {}'.format(Precision_H))
#     print('Recall: {}'.format(Recall_H))
#     print('F1: {}'.format(F1_H))
#     print('=======Prediabetic======')
#     print('Precision: {}'.format(Precision_P))
#     print('Recall: {}'.format(Recall_P))
#     print('F1: {}'.format(F1_P))
#     print('=======Diabetic======')
#     print('Precision: {}'.format(Precision_D))
#     print('Recall: {}'.format(Recall_D))
#     print('F1: {}'.format(F1_D))
#     print('======Overall======')
#     print('accuracy: {}'.format(acc))
#     print('balacned accuracy: {}'.format(bal_acc))

#     print(conf_mat_v2g)


#     print('=====CCA Weights=====')
#     print(weights_X)
#     print(weights_Y)
