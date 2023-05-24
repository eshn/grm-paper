import yaml
import numpy as np
import os

def getStats(dat):
    dat = np.array(dat)

    avg_mean = np.mean(dat)
    stdv = np.std(dat)
    ran = np.max(dat) - np.min(dat)

    return avg_mean, stdv, ran


def splitParams(ppath, outpath='./data/post-fit/'):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    with open(ppath, 'r') as f:
        ps = yaml.safe_load(f)
    f.close()

    data = ps['SUBJECTS']['DATA']

    participants = [data[i]['subject'] for i in range(len(data))]



    # for i in range(1):
    for i in range(len(data)):
        subj_name = data[i]['subject']
        avg = data[i]['avg']
        stdv = data[i]['stdv']

        peaks = data[i]['params']

        allA1 = []
        allA2 = []
        lam_list = []
        umax = []

        for c_peak in peaks: # peaks is a dictionary of all peaks, peaks[c_peak] is a dictionary containing fitted params for the peak "c_peak"
            allA1.append(peaks[c_peak]['A1'])
            allA2.append(peaks[c_peak]['A2'])
            lam_list.append(peaks[c_peak]['lam'])
            umax.append(max(peaks[c_peak]['u']))

        A1, A1stdv, A1range = getStats(allA1)
        A2, A2stdv, A2range = getStats(allA2)
        lam, lamstdv, lamrange = getStats(lam_list)
        
        umax_avg = np.mean(np.array(umax))

        outname = subj_name

         # dictionary of outputs
        dict_file = {'A1': A1.item(), 'A1range': A1range.item(), 'A1stdv': A1stdv.item(), 'A2': A2.item(), 'A2range': A2range.item(), 'A2stdv': A2stdv.item(), 'allA1': allA1,
        'allA2': allA2, 'avg': avg, 'lam': lam.item(), 'lam_list': lam_list, 'lamrange': lamrange.item(), 'lamstdv': lamstdv.item(), 'stdv': stdv, 'umax': umax, 'umax av': umax_avg.item()}

        # write to new yaml
        with open(outpath + data[i]['subject'] + '.yaml', 'w') as g:
            yaml.dump(dict_file, g, default_flow_style=False)
        g.close()
