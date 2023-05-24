import datamod as dm
import os
import yaml
from pathlib import Path

if __name__ == "__main__":
    
    import argparse
    from src.config import cfg


    ######################################################
    # Config Path
    ######################################################
    # Take experiment configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)

    args = parser.parse_args()
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # Set up the path of the folder containing the raw EXCELs and convert them to csv files
    excel_dir = Path(cfg.RAW_DATA_DIR)
    csv_dir = Path(cfg.GLUCOSE_DATA_DIR)

    excel_files = []

    for file in os.listdir(excel_dir):
        if file.endswith(".xlsx"):
            excel_files.append(os.path.join(excel_dir, file))

    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)
    
    print('This script will rewrite the csv files in', csv_dir, ", Press Y/y to confirm the conversion:") 
    rewrite = input()
    if rewrite in ['y', 'Y']:
        
        for excel_path in excel_files:
            dm.main(excel_path, csv_dir)
