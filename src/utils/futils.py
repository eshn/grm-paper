import pandas as pd
import os
import yaml

def get_data(fpath, studies='all', include_personal_measurements=True, include_diags_only=True):
    """
    File path to masterlist.csv. The columns are:
        0 - Study
        1 - Participant ID
        2 - Age
        3 - Sex
        4 - Height (m)
        5 - Weight (kg)
        6 - BMI (m/kg^2)
        7 - Systolic Blood Pressure
        8 - Diastolic Blood Pressure
        9 - Resting HR (bpm)
        10 - CANRISK
        11 - FBG
        12 - OGTT
        13 - HbA1c (%)
        14 - Alcohol
        15 - Meds
        16 - Group
        17 - Diagnosis
        18 - Class
    """

    # reads data
    data = pd.read_csv(fpath, delimiter=',', skiprows=[0])

    # get rid of columns irrelevant to the comparisons study (can be amended later).
    if include_personal_measurements:
        cols2drop = [7, 8, 9, 10, 14, 15, 16]
    else:
        cols2drop = [2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16]

    # get rid of the entries that is not part of the listed studies
    if studies != 'all':
        data = data.drop(data[~data['Study'].isin(studies)].index)

    if include_diags_only:
        # get rid of entries that don't have doctors eval
        data = data.dropna(subset = ['Diagnosis', 'Class'])
  
    return data


def load_params_from_yaml(params, data, yaml_path, include_diabetics=True, specify_sex='all', min_peaks=25):
    """
    Inputs:
        params - Parameters object to store all the values
        participants - List of participants
        yaml_path - path of yaml file to grab parameter values
    Steps:
        - opens yaml file specified in yaml_path
        - check that each subject in the yaml file has a corresponding entry in the masterlist
        - if its there, add everything to "params" object
        - if its not there, go to the next case
    Returns:
        params - Parameters object with data filled
        not_found - list of participants that have yaml files but not found in masterlist.csv
    """
    not_found = []
    not_enough_peaks = []
    ## Convert everything in the data frame to lists
    # particpant list
    participant_list = data['Participant ID'].to_list()

    # personal measurements
    age = data['Age'].to_list()
    sex = data['Sex'].to_list()
    height = data['Height (m)'].to_list()
    weight = data['Weight (kg)'].to_list()
    bmi = data['BMI (kg/m^2)'].to_list()

    # standardized metrics
    a1c = data['HbA1c (%)'].to_list()
    ogtt = data['OGTT'].to_list()
    fbg = data['FBG'].to_list()

    # dr's eval
    diags = data['Class'].to_list()

    ### opens yaml file specified in yaml_path
    # reads the list of yaml files
    flist = []
    for fname in os.listdir(yaml_path):
        if fname[-5:] == '.yaml':
            flist.append(fname)
    
    ### check that each subject in the yaml file has a corresponding entry in the masterlist    
    for id in flist: # iterates through participants
        if id[:-5] not in participant_list: ## if the yaml is not on participant list
            not_found.append(id[:-5])
        else:
            loc = participant_list.index(id[:-5]) # finds the index of the current subject in participant list

            if (not include_diabetics) and (diags[loc] == 'Diabetic'): #skips the diabetic cases if needed
                continue
            else:
                ### if its there, add everything to "params" object
                if specify_sex == 'all' or specify_sex == sex[loc]:
                    # add params to object
                    with open(yaml_path + id, 'r') as f:
                        ps = yaml.safe_load(f)
                    f.close()
                    if len(ps['allA1']) < min_peaks:
                        not_enough_peaks.append(id[:-5])
                        print(id[:-5], len(ps['allA1']))
                        continue
                    else:
                        params.A1.append(ps['A1'])
                        params.A1stdv.append(ps['A1stdv'])
                        params.A1range.append(ps['A1range'])
                        params.A2.append(ps['A2'])
                        params.A2stdv.append(ps['A2stdv'])
                        params.A2range.append(ps['A2range'])
                        params.allA1.append(ps['allA1'])
                        params.allA2.append(ps['allA2'])
                        params.avg.append(ps['avg'])
                        params.lam.append(ps['lam'])
                        params.lamstdv.append(ps['lamstdv'])
                        params.lamrange.append(ps['lamrange'])
                        params.lam_list.append(ps['lam_list'])
                        params.stdv.append(ps['stdv'])
                        params.umax.append(ps['umax'])
                        params.umax_av.append(ps['umax av'])

                        # add standardized metrics to object
                        params.a1c.append(a1c[loc])
                        params.ogtt.append(ogtt[loc])
                        params.fbg.append(fbg[loc])
                        
                        # add personal measurements to object
                        params.age.append(age[loc])
                        params.sex.append(sex[loc])
                        params.height.append(height[loc])
                        params.weight.append(weight[loc])
                        params.bmi.append(bmi[loc])

                        # add dr's eval, and the corresponding color to object
                        params.diags.append(diags[loc])
                        if diags[loc] == 'Diabetic':
                            params.colors.append('r')
                        elif diags[loc] == 'Prediabetic':
                            params.colors.append('y')
                        elif diags[loc] == 'Non Diabetic':
                            params.colors.append('g')
                        else:
                            params.colors.append('')

                        # add participant to object
                        params.participants.append(id[:-5])

                ### if its not there, go to the next case   
                else:
                    continue


    ### return the params object
    return params, not_found, not_enough_peaks

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


if __name__ == '__main__':
    fpath = '~/Documents/Glucose_Homeostasis/raw_data/masterlist.csv'

    # list of studies to keep
    studies = ['FOLLOW UP']

    data = get_data(fpath, studies)
