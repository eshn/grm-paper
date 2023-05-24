import os

from yacs.config import CfgNode as CN

_C = CN()
_C.SUBJECTS = CN(new_allowed=True)

def read_peak(peak_yaml):
    """
    The return data format is a dictionary, with its key to be subject index, and its value to be a dict of subject's data reading from peak.yaml.
    
    {
        1: {'subject':1, 'g_original': [0.544554455, 0.608360836, 0.706270627], 'max_val': 10.1, 'avg': 5.771729958, 'stdv': 0.833145439},
        2: {'subject':2, 'g_original': [0.544554455, 0.608360836, 0.706270627], 'max_val': 10.1, 'avg': 5.771729958, 'stdv': 0.833145439},
    }
    """
    cfg = _C
    cfg.defrost()
    cfg.merge_from_file(peak_yaml)
    cfg.freeze()
    data = cfg.SUBJECTS['DATA']
    data = {item['subject']:item for item in data}

    return data


if __name__ == '__main__':
    cfg = _C
 
    cfg.defrost()
 
    cfg.merge_from_file("clean_peak.yaml")
  
    cfg.freeze()

    #print(cfg.SUBJECTS['DATA'])
    print(read_peak("clean_peak.yaml"))