from src.utils.futils import *
import src.PeakAnalysis.peak_analysis as pa
import os
import yaml

## the imports below are strictly for plotting
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score


def get_scores_from_conf_mat(mat):
    total = np.sum(mat[:])
    ### compute TP, FP, FN, TB for each class
    TP_H = mat[0,0]
    TP_P = mat[1,1]
    TP_D = mat[2,2]

    # FP should be column sum without diag entry
    FP_H = np.sum(mat[:,0]) - TP_H
    FP_P = np.sum(mat[:,1]) - TP_P
    FP_D = np.sum(mat[:,2]) - TP_D

    # FN is row sum without diag entry
    FN_H = np.sum(mat[0,:]) - TP_H
    FN_P = np.sum(mat[1,:]) - TP_P
    FN_D = np.sum(mat[2,:]) - TP_D

    # TN is the sum of everywhere else, i.e. sum of all entries - TP - FP - FN
    TN_H = total - TP_H - FP_H - FN_H
    TN_P = total - TP_P - FP_P - FN_P
    TN_D = total - TP_D - FP_D - FN_D

    ### compute TNR and TPR for each class
    TPR_H = TP_H / (TP_H + FN_H)
    TNR_H = TN_H / (FP_H + TN_H)

    TPR_P = TP_P / (TP_P + FN_P)
    TNR_P = TN_P / (FP_P + TN_P)

    TPR_D = TP_D / (TP_D + FN_D)
    TNR_D = TN_D / (FP_D + TN_D)

    ### Compute precision/recall/F1 for each class
    Precision_H = TP_H / np.sum(mat[:,0])
    Recall_H = TP_H / np.sum(mat[0,:])
    F1_H = 2 * (Precision_H * Recall_H) / (Precision_H + Recall_H)

    Precision_P = TP_P / np.sum(mat[:,1])
    Recall_P = TP_P / np.sum(mat[1,:])
    F1_P = 2 * (Precision_P * Recall_P) / (Precision_P + Recall_P)

    Precision_D = TP_D / np.sum(mat[:,2])
    Recall_D = TP_D / np.sum(mat[2,:])
    F1_D = 2 * (Precision_D * Recall_D) / (Precision_D + Recall_D)

    ### compute accuracy and balanced accuracy
    acc = np.trace(mat) / total
    bal_acc = (TPR_H + TPR_P + TPR_D) / 3

    precisions = [Precision_H, Precision_P, Precision_D]
    recalls = [Recall_H, Recall_P, Recall_D]
    F1 = [F1_H, F1_P, F1_D]
    accuracies = [acc, bal_acc]

    return precisions, recalls, F1, accuracies

class Parameters():
    def __init__(self):
        ############## read directly from file
        # model parameters from yaml
        self.A1 = []
        self.A1stdv = []
        self.A1range = []
        self.A2 = []
        self.A2stdv = []
        self.A2range = []
        self.allA1 = []
        self.allA2 = []
        self.avg = []
        self.lam = []
        self.lamstdv = []
        self.lamrange = []
        self.lam_list = []
        self.stdv = []
        self.umax = []
        self.umax_av = []

        # standardized metrics
        self.a1c = []
        self.ogtt = []
        self.fbg = []

        # personal measurements
        self.age = []
        self.sex = []
        self.height = []
        self.weight = []
        self.bmi = []
        self.bp_sys = []
        self.bp_dia = []
        self.hr = []

        # doctor's eval
        self.diags = []
        self.colors = [] # color based on dr's eval used for plotting purposes

        # partcipant ID
        self.participants = []

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
                help='Specifies mode which mode to run. 1: Train Only. 2: Evaluate with Weights from Paper (or specify your own). 3: Train+Evaluate (uses included data unless otherwise specified)',
                required=False,
                type=int,
                default=4)
    parser.add_argument('--mlist',
                help='File path to participant_list.csv',
                required=False,
                type=str,
                default='./data/participant_list.csv')
    parser.add_argument('--trainstudy',
                help='Study name of training set listed in participant_list.csv',
                required=False,
                type=str,
                default='KLICK150')
    parser.add_argument('--teststudy',
                help='Study name of test set listed in participant_list.csv',
                required=False,
                type=str,
                default='KLICK600')                
    parser.add_argument('--trainset',
                help='Folder path to training data',
                required=False,
                type=str,
                default='./data/post-fit_params/batch_params_klick150/')
    parser.add_argument('--testset',
                help='Folder path to test data',
                required=False,
                type=str,
                default='./data/post-fit_params/batch_params_klick600/')
    args = parser.parse_args()

    if args.mode == 3:
        masterlist = args.mlist

        # path to yaml files containing model parameters after fitting to model. can be a list of paths if there are multiple folders
        train_data = [args.trainset]
        test_data = [args.testset]

        # initialize objects to store model parameters
        p_train = Parameters()
        p_test = Parameters()

        # Load subject data
        print('Loading Subject Data From ' + masterlist)
        data_train = get_data(fpath=masterlist, studies=[args.trainstudy], include_personal_measurements=True, include_diags_only=False)
        data_test = get_data(fpath=masterlist, studies=[args.teststudy], include_personal_measurements=True, include_diags_only=False)

        # Drops individuals who are missing any measurements specified in the "filters" list
        filters = ['HbA1c (%)', 'Age', 'BMI (kg/m^2)', 'Class']
        data_train.dropna(subset=filters, inplace=True)
        data_test.dropna(subset=filters, inplace=True)

        # Load parameters from yaml files. Passes in an object from the Parameters class then returns the same object but filled
        print('Reading Parameter Data from YAML...')
        for ypath in train_data:
            p_train, _, _ = load_params_from_yaml(params=p_train, data=data_train, yaml_path=ypath)
        for ypath in test_data:
            p_test, _, _ = load_params_from_yaml(params=p_test, data=data_test, yaml_path=ypath)

        thresholds, weights, train_params = pa.train(p_train)
        # train_params is a list consisting of the following:
        # train_params[0]: list containing two elements: 1) list of normalization factors in X (index 0), and 2) normalization factor in Y (scalar)
        # train_params[1]: list of two elements: 1) column indices of X that are exponentially augmented, 2) the corresponding exp growth/decay rate


        GRM_train, pred_train, labels_train, conf_mat_train = pa.predict(p_train, thresholds, weights, train_params)
        GRM_test, pred_test, labels_test, conf_mat_test = pa.predict(p_test, thresholds, weights, train_params)


        #### PLOTTING AND SUMMARY STATISTICS
        fig, ax = plt.subplots(1, 2)

        ''' PREDICTIONS WITH TRAINING SET'''
        prec, rec, f1, accs = get_scores_from_conf_mat(conf_mat_train)

        #### OvR:
        # split into three sets for each OvR classification: healthy vs rest, prediabetic vs rest, diabetic vs rest
        h_label = np.array([1 if i == 1 else 0 for i in labels_train])
        p_label = np.array([1 if i == 2 else 0 for i in labels_train])
        d_label = np.array([1 if i == 3 else 0 for i in labels_train])

        # flip the order so healthy is closer to 1
        h_pred = GRM_train / np.max(GRM_train)
        p_pred = copy(h_pred)
        d_pred = copy(h_pred)

        # flip the order so healthy is closer to 1
        h_pred = np.ones(h_pred.shape) - h_pred


        h_fpr, h_tpr, h_thresholds = roc_curve(h_label, h_pred)
        p_fpr, p_tpr, p_thresholds = roc_curve(p_label, p_pred)
        d_fpr, d_tpr, d_thresholds = roc_curve(d_label, d_pred)

        h_prec, h_rec, _ = precision_recall_curve(h_label, h_pred)
        p_prec, p_rec, _ = precision_recall_curve(p_label, p_pred)
        d_prec, d_rec, _ = precision_recall_curve(d_label, d_pred)


        ax[0].plot(h_fpr, h_tpr, label='Non-Diabetic. AUC: {:.4f}'.format(roc_auc_score(h_label, h_pred)))
        # ax[0].plot(p_fpr, p_tpr, label='Prediabetic. AUC: {:.4f}'.format(roc_auc_score(p_label, p_pred)))
        ax[0].plot(d_fpr, d_tpr, label='Diabetic. AUC: {:.4f}'.format(roc_auc_score(d_label, d_pred)))
        ax[0].plot([0,1], [0,1], '--')
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_title('ROC for Klick-150')
        ax[0].grid(alpha=0.4)
        ax[0].legend(loc='lower right')

        Precision_H = prec[0]
        Precision_P = prec[1]
        Precision_D = prec[2]

        Recall_H = rec[0]
        Recall_P = rec[1]
        Recall_D = rec[2]

        F1_H = f1[0]
        F1_P = f1[1]
        F1_D = f1[2]

        acc = accs[0]
        bal_acc = accs[1]

        print('-------------Klick150-------------')
        print('Total Predictions: {}'.format(int(np.sum(conf_mat_train[:]))))
        print('Males: {}'.format(np.sum(np.array([i == 'Male' for i in p_train.sex]))))
        print('Females: {}'.format(np.sum(np.array([i == 'Female' for i in p_train.sex]))))
        print('=======Healthy======')
        print('Precision: {}'.format(Precision_H))
        print('Recall: {}'.format(Recall_H))
        print('F1: {}'.format(F1_H))
        print('=======Prediabetic======')
        print('Precision: {}'.format(Precision_P))
        print('Recall: {}'.format(Recall_P))
        print('F1: {}'.format(F1_P))
        print('=======Diabetic======')
        print('Precision: {}'.format(Precision_D))
        print('Recall: {}'.format(Recall_D))
        print('F1: {}'.format(F1_D))
        print('========ROCs=======')
        print('healthy roc auc: {}'.format(roc_auc_score(h_label, h_pred)))
        # print('prediabetic roc auc: {}'.format(roc_auc_score(p_label, p_pred)))
        print('diabetic roc auc: {}'.format(roc_auc_score(d_label, d_pred)))
        print('======Overall======')
        print('accuracy: {}'.format(acc))
        print('balacned accuracy: {}'.format(bal_acc))

        print(conf_mat_train)


        ''' PREDICTIONS WITH TEST SET '''
        prec, rec, f1, accs = get_scores_from_conf_mat(conf_mat_test)
        #### OvR:
        # split into three sets for each OvR classification: healthy vs rest, prediabetic vs rest, diabetic vs rest
        h_label = np.array([1 if i == 1 else 0 for i in labels_test])
        p_label = np.array([1 if i == 2 else 0 for i in labels_test])
        d_label = np.array([1 if i == 3 else 0 for i in labels_test])

        # flip the order so healthy is closer to 1
        h_pred = GRM_test / np.max(GRM_test)
        p_pred = copy(h_pred)
        d_pred = copy(h_pred)

        # flip the order so healthy is closer to 1
        h_pred = np.ones(h_pred.shape) - h_pred

        h_fpr, h_tpr, h_thresholds = roc_curve(h_label, h_pred)
        p_fpr, p_tpr, p_thresholds = roc_curve(p_label, p_pred)
        d_fpr, d_tpr, d_thresholds = roc_curve(d_label, d_pred)

        h_prec, h_rec, _ = precision_recall_curve(h_label, h_pred)
        p_prec, p_rec, _ = precision_recall_curve(p_label, p_pred)
        d_prec, d_rec, _ = precision_recall_curve(d_label, d_pred)


        print('healthy roc auc: {}'.format(roc_auc_score(h_label, h_pred)))
        # print('prediabetic roc auc: {}'.format(roc_auc_score(p_label, p_pred)))
        print('diabetic roc auc: {}'.format(roc_auc_score(d_label, d_pred)))

        ax[1].plot(h_fpr, h_tpr, label='Non-Diabetic. AUC: {:.4f}'.format(roc_auc_score(h_label, h_pred)))
        # ax[1].plot(p_fpr, p_tpr, label='Prediabetic. AUC: {:.4f}'.format(roc_auc_score(p_label, p_pred)))
        ax[1].plot(d_fpr, d_tpr, label='Diabetic. AUC: {:.4f}'.format(roc_auc_score(d_label, d_pred)))
        ax[1].plot([0,1], [0,1], '--')
        ax[1].set_xlabel('False Positive Rate')
        ax[1].set_ylabel('True Positive Rate')
        ax[1].set_title('ROC for Klick-600')
        ax[1].grid(alpha=0.4)
        ax[1].legend(loc='lower right')

        Precision_H = prec[0]
        Precision_P = prec[1]
        Precision_D = prec[2]

        Recall_H = rec[0]
        Recall_P = rec[1]
        Recall_D = rec[2]

        F1_H = f1[0]
        F1_P = f1[1]
        F1_D = f1[2]

        acc = accs[0]
        bal_acc = accs[1]

        print('-------------Klick600-------------')
        print('Total Predictions: {}'.format(int(np.sum(conf_mat_test[:]))))
        print('Males: {}'.format(np.sum(np.array([i == 'Male' for i in p_test.sex]))))
        print('Females: {}'.format(np.sum(np.array([i == 'Female' for i in p_test.sex]))))
        print('=======Healthy======')
        print('Precision: {}'.format(Precision_H))
        print('Recall: {}'.format(Recall_H))
        print('F1: {}'.format(F1_H))
        print('=======Prediabetic======')
        print('Precision: {}'.format(Precision_P))
        print('Recall: {}'.format(Recall_P))
        print('F1: {}'.format(F1_P))
        print('=======Diabetic======')
        print('Precision: {}'.format(Precision_D))
        print('Recall: {}'.format(Recall_D))
        print('F1: {}'.format(F1_D))
        print('========ROCs=======')
        print('healthy roc auc: {}'.format(roc_auc_score(h_label, h_pred)))
        # print('prediabetic roc auc: {}'.format(roc_auc_score(p_label, p_pred)))
        print('diabetic roc auc: {}'.format(roc_auc_score(d_label, d_pred)))
        print('======Overall======')
        print('accuracy: {}'.format(acc))
        print('balacned accuracy: {}'.format(bal_acc))

        print(conf_mat_test)

        plt.show()

        print('--------THRESHOLDS, WEIGHTS, AND TRAINING PARAMS------')
        print('Thresholds: {}'.format(thresholds))
        print('Weights: {}'.format(weights))
        print('Exponential Params: {}'.format(train_params))

    elif args.mode == 2:
        masterlist = args.mlist

        # path to yaml files containing model parameters after fitting to model. can be a list of paths if there are multiple folders
        test_data = [args.testset]

        # initialize objects to store model parameters
        p_test = Parameters()

        # Load subject data
        print('Loading Subject Data From ' + masterlist)
        data_test = get_data(fpath=masterlist, studies=[args.teststudy], include_personal_measurements=True, include_diags_only=False)

        # Drops individuals who are missing any measurements specified in the "filters" list
        filters = ['HbA1c (%)', 'Age', 'BMI (kg/m^2)', 'Class']
        data_test.dropna(subset=filters, inplace=True)

        # Load parameters from yaml files. Passes in an object from the Parameters class then returns the same object but filled
        print('Reading Parameter Data from YAML...')
        for ypath in test_data:
            p_test, _, _ = load_params_from_yaml(params=p_test, data=data_test, yaml_path=ypath)


        # weights from paper
        thresholds = [0.48437462633121703, 0.5630536488964719]
        # thresholds = [0.45837462633121703, 0.5630536488964719]
        weightsX = np.array([[-0.01729736],
       [-0.00124415],
       [-0.00186463],
       [ 0.00773906],
       [ 0.02451672],
       [-0.01195743],
       [-0.00032777],
       [-0.07090241],
       [ 0.03363849],
       [-0.01194681],
       [-0.00919847],
       [-0.0074819 ],
       [-0.00141135]])
        weightsY = np.array([[-0.13321868]])

        weights = [weightsX, weightsY]

        train_params = [[np.array([ 0.20933155,  0.15639031,  0.68704591,  0.09740014,  0.09234949,
        0.69248676, 21.43987957,  0.50141083,  0.14864092,  0.89010111,
        3.87393704, 60.        , 43.3       ]), 11.9], [[0, 6, 10], [-3.787044, 4.423684, 1.355863]]]


        GRM_test, pred_test, labels_test, conf_mat_test = pa.predict(p_test, thresholds, weights, train_params)


        #### PLOTTING AND SUMMARY STATISTICs
        ''' PREDICTIONS WITH TEST SET '''
        prec, rec, f1, accs = get_scores_from_conf_mat(conf_mat_test)
        #### OvR:
        # split into three sets for each OvR classification: healthy vs rest, prediabetic vs rest, diabetic vs rest
        h_label = np.array([1 if i == 1 else 0 for i in labels_test])
        p_label = np.array([1 if i == 2 else 0 for i in labels_test])
        d_label = np.array([1 if i == 3 else 0 for i in labels_test])

        # flip the order so healthy is closer to 1
        h_pred = GRM_test / np.max(GRM_test)
        p_pred = copy(h_pred)
        d_pred = copy(h_pred)

        # flip the order so healthy is closer to 1
        h_pred = np.ones(h_pred.shape) - h_pred

        h_fpr, h_tpr, h_thresholds = roc_curve(h_label, h_pred)
        p_fpr, p_tpr, p_thresholds = roc_curve(p_label, p_pred)
        d_fpr, d_tpr, d_thresholds = roc_curve(d_label, d_pred)

        h_prec, h_rec, _ = precision_recall_curve(h_label, h_pred)
        p_prec, p_rec, _ = precision_recall_curve(p_label, p_pred)
        d_prec, d_rec, _ = precision_recall_curve(d_label, d_pred)


        print('healthy roc auc: {}'.format(roc_auc_score(h_label, h_pred)))
        # print('prediabetic roc auc: {}'.format(roc_auc_score(p_label, p_pred)))
        print('diabetic roc auc: {}'.format(roc_auc_score(d_label, d_pred)))

        fig, ax = plt.subplots(1)
        ax.plot(h_fpr, h_tpr, label='Non-Diabetic. AUC: {:.4f}'.format(roc_auc_score(h_label, h_pred)))
        # ax[1].plot(p_fpr, p_tpr, label='Prediabetic. AUC: {:.4f}'.format(roc_auc_score(p_label, p_pred)))
        ax.plot(d_fpr, d_tpr, label='Diabetic. AUC: {:.4f}'.format(roc_auc_score(d_label, d_pred)))
        ax.plot([0,1], [0,1], '--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
        ax.grid(alpha=0.4)
        ax.legend(loc='lower right')

        Precision_H = prec[0]
        Precision_P = prec[1]
        Precision_D = prec[2]

        Recall_H = rec[0]
        Recall_P = rec[1]
        Recall_D = rec[2]

        F1_H = f1[0]
        F1_P = f1[1]
        F1_D = f1[2]

        acc = accs[0]
        bal_acc = accs[1]

        print('-------------Statistics-------------')
        print('Total Predictions: {}'.format(int(np.sum(conf_mat_test[:]))))
        print('Males: {}'.format(np.sum(np.array([i == 'Male' for i in p_test.sex]))))
        print('Females: {}'.format(np.sum(np.array([i == 'Female' for i in p_test.sex]))))
        print('=======Healthy======')
        print('Precision: {}'.format(Precision_H))
        print('Recall: {}'.format(Recall_H))
        print('F1: {}'.format(F1_H))
        print('=======Prediabetic======')
        print('Precision: {}'.format(Precision_P))
        print('Recall: {}'.format(Recall_P))
        print('F1: {}'.format(F1_P))
        print('=======Diabetic======')
        print('Precision: {}'.format(Precision_D))
        print('Recall: {}'.format(Recall_D))
        print('F1: {}'.format(F1_D))
        print('========ROCs=======')
        print('healthy roc auc: {}'.format(roc_auc_score(h_label, h_pred)))
        # print('prediabetic roc auc: {}'.format(roc_auc_score(p_label, p_pred)))
        print('diabetic roc auc: {}'.format(roc_auc_score(d_label, d_pred)))
        print('======Overall======')
        print('accuracy: {}'.format(acc))
        print('balacned accuracy: {}'.format(bal_acc))

        print(conf_mat_test)

        plt.show()
    elif args.mode == 1:
        masterlist = args.mlist

        # path to yaml files containing model parameters after fitting to model. can be a list of paths if there are multiple folders
        train_data = [args.trainset]

        # initialize objects to store model parameters
        p_train = Parameters()

        # Load subject data
        print('Loading Subject Data From ' + masterlist)
        data_train = get_data(fpath=masterlist, studies=[args.trainstudy], include_personal_measurements=True, include_diags_only=False)

        # Drops individuals who are missing any measurements specified in the "filters" list
        filters = ['HbA1c (%)', 'Age', 'BMI (kg/m^2)', 'Class']
        data_train.dropna(subset=filters, inplace=True)

        # Load parameters from yaml files. Passes in an object from the Parameters class then returns the same object but filled
        print('Reading Parameter Data from YAML...')
        for ypath in train_data:
            p_train, _, _ = load_params_from_yaml(params=p_train, data=data_train, yaml_path=ypath)

        thresholds, weights, train_params = pa.train(p_train)

        GRM_train, pred_train, labels_train, conf_mat_train = pa.predict(p_train, thresholds, weights, train_params)

        #### PLOTTING AND SUMMARY STATISTICS
        fig, ax = plt.subplots(1)

        ''' PREDICTIONS WITH TRAINING SET'''
        prec, rec, f1, accs = get_scores_from_conf_mat(conf_mat_train)

        #### OvR:
        # split into three sets for each OvR classification: healthy vs rest, prediabetic vs rest, diabetic vs rest
        h_label = np.array([1 if i == 1 else 0 for i in labels_train])
        p_label = np.array([1 if i == 2 else 0 for i in labels_train])
        d_label = np.array([1 if i == 3 else 0 for i in labels_train])

        # flip the order so healthy is closer to 1
        h_pred = GRM_train / np.max(GRM_train)
        p_pred = copy(h_pred)
        d_pred = copy(h_pred)

        # flip the order so healthy is closer to 1
        h_pred = np.ones(h_pred.shape) - h_pred


        h_fpr, h_tpr, h_thresholds = roc_curve(h_label, h_pred)
        p_fpr, p_tpr, p_thresholds = roc_curve(p_label, p_pred)
        d_fpr, d_tpr, d_thresholds = roc_curve(d_label, d_pred)

        h_prec, h_rec, _ = precision_recall_curve(h_label, h_pred)
        p_prec, p_rec, _ = precision_recall_curve(p_label, p_pred)
        d_prec, d_rec, _ = precision_recall_curve(d_label, d_pred)


        ax.plot(h_fpr, h_tpr, label='Non-Diabetic. AUC: {:.4f}'.format(roc_auc_score(h_label, h_pred)))
        # ax.plot(p_fpr, p_tpr, label='Prediabetic. AUC: {:.4f}'.format(roc_auc_score(p_label, p_pred)))
        ax.plot(d_fpr, d_tpr, label='Diabetic. AUC: {:.4f}'.format(roc_auc_score(d_label, d_pred)))
        ax.plot([0,1], [0,1], '--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC for Klick-150')
        ax.grid(alpha=0.4)
        ax.legend(loc='lower right')

        Precision_H = prec[0]
        Precision_P = prec[1]
        Precision_D = prec[2]

        Recall_H = rec[0]
        Recall_P = rec[1]
        Recall_D = rec[2]

        F1_H = f1[0]
        F1_P = f1[1]
        F1_D = f1[2]

        acc = accs[0]
        bal_acc = accs[1]

        print('-------------Summary-------------')
        print('Total Predictions: {}'.format(int(np.sum(conf_mat_train[:]))))
        print('Males: {}'.format(np.sum(np.array([i == 'Male' for i in p_train.sex]))))
        print('Females: {}'.format(np.sum(np.array([i == 'Female' for i in p_train.sex]))))
        print('=======Healthy======')
        print('Precision: {}'.format(Precision_H))
        print('Recall: {}'.format(Recall_H))
        print('F1: {}'.format(F1_H))
        print('=======Prediabetic======')
        print('Precision: {}'.format(Precision_P))
        print('Recall: {}'.format(Recall_P))
        print('F1: {}'.format(F1_P))
        print('=======Diabetic======')
        print('Precision: {}'.format(Precision_D))
        print('Recall: {}'.format(Recall_D))
        print('F1: {}'.format(F1_D))
        print('========ROCs=======')
        print('healthy roc auc: {}'.format(roc_auc_score(h_label, h_pred)))
        # print('prediabetic roc auc: {}'.format(roc_auc_score(p_label, p_pred)))
        print('diabetic roc auc: {}'.format(roc_auc_score(d_label, d_pred)))
        print('======Overall======')
        print('accuracy: {}'.format(acc))
        print('balacned accuracy: {}'.format(bal_acc))

        print(conf_mat_train)

        plt.show()

        print('--------THRESHOLDS, WEIGHTS, AND TRAINING PARAMS------')
        print('Thresholds: {}'.format(thresholds))
        print('Weights: {}'.format(weights))
        print('Exponential Params: {}'.format(train_params))