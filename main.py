import src.PeakSelection.peakSelection as ps
import src.GlucoseModel.subjectFitToModel as stm
from src.utils.splitModelParams import splitParams
import argparse
from src.config import cfg
import os
from pathlib import Path
import yaml


'''
Note that any data preprocessing work is not included. The script here assumes that the data is already cleaned
and stored in a CSV format. A sample of the CSV formatting is provided in [insert file path] 

This file should be executed with the following command:

    python3 main.py --cfg=[location]

where [location] is the path to the configuration yaml file. The --cfg switch is optional
and defaults to ./experiments/experiment.yaml
'''

# Sets up argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cfg',
                help='experiment configure file name',
                required=False,
                type=str,
                default='experiments/experiment.yaml')


args = parser.parse_args()
cfg.defrost()
cfg.merge_from_file(args.cfg)
cfg.freeze()

 # Set up peak yaml foler to save extracted peaks
peak_dir = Path(cfg.PEAK_DATA_DIR)
peak_yaml_filename = Path(cfg.PEAK_DATA_FILE)

###### runs peak selection on a batch of participants
ps.batch_PS()

###### Fit Extract Peaks to Model via Gradient Descent
stm.batch_FitToModel()
### split peak data into individual files
fname = os.path.basename(args.cfg)[:-5]
param_path = str(Path(cfg.LOG_DIR)) + '/' + fname + '/' + fname + '_peak/params.yaml'

splitParams(ppath=param_path, outpath='./data/post-fit/')