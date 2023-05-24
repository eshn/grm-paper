import pandas as pd
import csv
import os

def excel_to_df(excel_filename):
    df = pd.read_excel(excel_filename, sheet_name="Sheet1")
    
    #!------------Reassign eader------------------------------!
    if any('Unnamed' in s for s in df.columns):
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])
        df = df.reset_index(drop=True)
    
    #!------------Check reuquired data------------------------!
    assert all(x in df.columns for x in ['Time', 'Historic Glucose (mmol/L)', 'Scan Glucose (mmol/L)']),  \
        "File format error. Please ensure 'Time', 'Historic Glucose (mmol/L)', and 'Scan Glucose (mmol/L)' exist."

    #!------------Keep only required columns------------------!
    df = pd.concat([df['Time'], df['Historic Glucose (mmol/L)'], df['Scan Glucose (mmol/L)']], axis=1, sort=False)
    
    print(df)
    #df.to_csv('your_csv_file.csv', index = False)
    return df


def process_to_csv(df, csvFileName):

    #!------------Convert time--------------------------------!
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values(by=['Time'])

    #!------------Create Difference with Start Time-----------!
    df['delta'] = df['Time'] - df['Time'].iloc[0]

    #!------------Get Elapsed Time---------------------------!
    df['Elapsed Time'] = df['delta'].dt.total_seconds()

    del df['Time']
    del df['delta']

    #!------------Rearrange Columns---------------------------!
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]

    df = df[cols]

    #!------------Combines Glucose----------------------------!
    df['Glucose (mmol/L)'] = df['Historic Glucose (mmol/L)'].fillna(df['Scan Glucose (mmol/L)'])

    del df['Historic Glucose (mmol/L)']
    del df['Scan Glucose (mmol/L)']

    df = df.sort_values(by=['Elapsed Time'])

    #!------------Export to CSV-------------------------------!
    df.to_csv(csvFileName, index=False)


def main(excel_path, csv_dir):
    #READ EXCEL-------------------------------------------------------
    df = excel_to_df(excel_path)
    _csvFileName = os.path.splitext(os.path.basename(excel_path))[0] +  '.csv'
    _csvFilePath = os.path.join(csv_dir, _csvFileName)
    process_to_csv(df, _csvFilePath)



if __name__ == "__main__":
    """
    usage:
        python datamod.py <path-to-the-excel-file>
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('excelPath', help='path of the EXCEL file')
    parser.add_argument('csvDir', help='path of which directory the converted CSV will save to.')
    args = parser.parse_args()
    
    main(args.excelPath, args.csvDir)
