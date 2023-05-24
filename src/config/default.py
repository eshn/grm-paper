import os

from yacs.config import CfgNode as CN

_C = CN()

_C.RAW_DATA_DIR = ''
_C.GLUCOSE_DATA_DIR = ''
_C.PEAK_DATA_DIR = ''
_C.PEAK_DATA_FILE = ''
_C.LOG_DIR = ''

_C.EXP_INITIATE_FROM = 'RAW_DATA'

_C.RUN_SUBJECTS = CN()
### One Of The following options ###
_C.RUN_SUBJECTS.OPTIONS = 'ALL'
## 1. Run All ##
_C.RUN_SUBJECTS.ALL = True
## 2. Run Selected Subjects ##
_C.RUN_SUBJECTS.SELECTION = CN(new_allowed=True)

_C.PARAMS = CN()
_C.PARAMS.tstep = 1.0
_C.PARAMS.lam = 0.4116421339794699
_C.PARAMS.A1 = 0.1
_C.PARAMS.A2 = 0.1
_C.PARAMS.A3 = 0.005
_C.PARAMS.A4 = 1.0
_C.PARAMS.h = 1.0
_C.PARAMS.amp = 1.7524890366951094
_C.PARAMS.sig = 1.0
_C.PARAMS.delta = 0.01
_C.PARAMS.sc = 0.0001
_C.PARAMS.sc_min = 1e-12
_C.PARAMS.numIterations = 150000


if __name__ == '__main__':
    cfg = _C
 
    cfg.defrost()
 
    cfg.merge_from_file(os.path.join("../experiments", "some_user_define_experiement_name.yaml"))

    cfg.freeze()
    
    print(cfg)
