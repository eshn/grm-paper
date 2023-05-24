import csv
import json
import math
import matplotlib.axes as ax
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import time
import yaml
from copy import copy
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter1d
import src.PeakSelection.findPeak as fp
import src.utils.utils as ut

class PeakSelection:
    
    def __init__(self, csv_path, log_dir, data_cleanse_plot=True):
    
        self.csv_path = csv_path
        self.log_dir = log_dir
        self.subject_id = os.path.basename(csv_path).split('.')[0]

        print("—————————————————————————————————————————————————————")    
        print("Start clearning %s..." % (self.subject_id))
        print("—————————————————————————————————————————————————————\n")  

        self.glucose_df = self._read_glucose_from_csv(self.csv_path)
        self.clean_glucose_df = self._check_data_corruption(self.glucose_df, ['Glucose (mmol/L)'], force_keep_first = True)
        self.clean_glucose_df.dropna(inplace=True)
        self.total_time = max(self.clean_glucose_df.index.tolist())

        self.mean = np.mean(self.clean_glucose_df['Glucose (mmol/L)'].dropna())
        self.stdv = np.std(self.clean_glucose_df['Glucose (mmol/L)'].dropna())

        self.regular_splines = self._clean_irregularities(self.clean_glucose_df)
        self.merged_splines = self._link_dfs_within_buffer(self.regular_splines)

        print("\n-----------------------------------------------------\n")
        print("Finished Data Cleaning and spline Extraction.")
        print("\n-----------------------------------------------------\n")

        print("\n-----------------------------------------------------\n")
        print("Searching if manual selection peaks found in file.")
        print("\n-----------------------------------------------------\n")

        self.manual_peaks = self._manual_peaks()

        print("\n-----------------------------------------------------\n")
        print("Initialization done! Can start extracting peaks now.")
        print("\n-----------------------------------------------------\n")

        # This attributes will be contructed once run the peak extraction methods.
        # Format: {baseline_mode: {'peaks' : peaks, 'number_of_peaks' : len(peaks), 'Gaussian_smooth' : gaussian_stdv }}
        self.peaks_dict = {}

        # This attribute will be contructed once run 'top_peak_extraction' method.
        self.top_peaks = []

        if data_cleanse_plot == True:
            self.plot_original_vs_cleaned_data(save=True)


    def _read_glucose_from_csv(self, csv):
        """
        Read in csv as a pandas dataframe
        """
        glucose_df = pd.read_csv(csv)
        glucose_df = glucose_df.loc[:, ['Elapsed Time', 'Glucose (mmol/L)']]
        glucose_df['Elapsed Time'] = glucose_df['Elapsed Time'].astype(int)
        glucose_df = glucose_df.set_index('Elapsed Time')

        return glucose_df


    def _clean_duplicates(self, df, clean_NaN_from_cols=[]):
        """
        This function is to remove rows which have duplicate time index, and at the same time, either: 
        (1) have NaN value in column values - missing data
        (2) duplicated values - duplicated records
        The cleaning only applies on rows which have duplicate time index.

        If pass on multiple columns in 'clean_NaN_from_cols', it only removes the row when ALL the columns
        specified in 'clean_NaN_from_cols' have NaN value.

        args:
            - df:                   The target dataframe
            - clean_NaN_from_cols:  A list of column(s) name, from which the redundant data is to be cleaned.
                                    Default is all columns. 
        return:
            - df:                   The clean dataframe
        """
        df = df.copy()
        
        col_names = df.columns
        if (type(clean_NaN_from_cols)==list) & (len(clean_NaN_from_cols)>0):
            col_names = clean_NaN_from_cols
        
        print("\n-----------------------------------------------------\n")
        print("Cleaning the duplicate data:")
        print("\n-----------------------------------------------------\n")
        duplicated_idx = df.loc[df.index.duplicated(keep=False)]
        print("This File has duplicated Index:", duplicated_idx)
        
        # 1/ Clean the duplicated data with the same index
        df['index'] = df.index
        df = df.loc[~df.duplicated(keep='first')]
    
        # 2/ Clean the NaN row with repeated index
        df = df.loc[~df.index.duplicated(keep=False) | df[col_names].notnull().any(axis=1)]
        df.drop(columns=['index'], inplace=True)

        print("\n-----------------------------------------------------\n")
        print("Done cleaning duplicate data. Clean dataframe:")
        print("\n-----------------------------------------------------\n")
        print(df)

        return df


    def _check_data_corruption(self, df, clean_NaN_from_cols=[], force_keep_first = False):
        """
        args:
            - df:                   The target dataframe to be cleaned.
            - clean_NaN_from_cols:  The list of names of the column(s) which have redundant data to be cleaned.
                                    The redundent data are those rows which have duplicated indices with the
                                    other row(s), however its own value is either NaN or identical.
                                    Default is to check all columns.
            - force_keep_first:     If the dataframe has the duplicated
        return:
            df:                     If the data passes the check, it returns the df.
        
        This function will first clean up the redundant data (either empty or duplicated data) in the dataframe,
        and check if any error data remained, i.e., same indices (i.e., time) but with different (glucose) values.
        It raises 'Data corruption' exception error if such error is found. 

        The optional 'force_keep_first' flag ignores the 'Data Corruption' warning, and from the inconsistent
        data points (which have duplicated indice), it picks whichever encountered first, and drop the rest.
        """

        df = df.copy()
        df = self._clean_duplicates(df, clean_NaN_from_cols)

        if force_keep_first == True:

            df = df.loc[~df.index.duplicated(keep='first')].copy()
            return df

        if any(df.index.duplicated(keep=False)):
            # Raise Error
            print("\n-----------------------------------------------------\n")
            print("Data Corruption: The dataset has duplicated time index but with different glucose values.")
            print(df.loc[df.index.duplicated(keep=False)])
            print("\n-----------------------------------------------------\n")
            raise Exception("Data corruption: The dataset has duplicated time index but with different glucose values.")


    def _clean_irregularities(self, df, idx_delta=900, min_datapoints=4, log=False):
        """
        args:
            df:             A dataframe from which regular data will be extracted. 
            idx_delta:      The index interval to be used to extract the regular data section.      
            min_datapoints: The minumum number of data points in the regular data section. Any section with
                            data points lesser than this number would be dropped.
    
        returns:
            reg_splines_list: 
                            A list of dataframes, which are regular data sections (i.e., with fixed index interval
                            and continuous data points). The list is sorted based on the start idex of each dataframe.

        Given a clean dataframe of glucose data (with no duplicated time index and NaN (missing) values),
        this function would extract regular and continuous data which have fixed intervals of delta time
        (i.e., 900 seconds) between successive data points. 
        For example, a regular section with a uniformed delta interval (i.e., 900 sec) between successive index
        (i.e., Elapsed Time) is:
        
        e.g., 
                'Elapsed Time'  'Glucose (mmol/L)'
                960             5.4   
                1860             6.3
                2760             7.6
                .... 

        Note:   One dataframe can have multiple regular splines. This function would extract any regular
                spline that has a continuous delta interval (i.e., 900 sec), and only extract those that
                have sufficient data points (>= min_datapoints).

                For example, a dataframe can have the following Time index, with 'min_datapoints' = 3:
                    [0, 900, 960, 1000, 1800, 1860, 2000, 2700, 5000, 5900, 6800 ....]
                
                    1. 0 - 900 - 1800 - 2700    [Selected]
                    2. 960 - 1860               [Ignore]
                    3. 5000 - 5900 - 6800       [Selected]
        """

        print("\n-----------------------------------------------------\n")
        print("Start clean up irregularities:")
        print("\n-----------------------------------------------------\n")
        print("Each glucose data point should be collected by a regular 15 minutes interval. \n")
        print("The regular-interval splines (glucose series) is extracted into separate dataframes,\n")
        print("and sorted in a list by its start time.\n")

        df = df.copy()
        reg_splines_list = []

        visited_start = []
        
        for cur_time in sorted(df.index):

            # Check if the current time stamp is a subsequent node of the visited antecedents.
            precedents = [((cur_time - s) % idx_delta == 0) for s in visited_start]

            if not any(precedents):
                visited_start.append(cur_time)

                # Select all index satisfied: (cur_time + n * 900)
                idx_seq = [idx for idx in df.index if ((idx-cur_time) % idx_delta)==0]
                spline = df.loc[idx_seq].copy()

                # Check breaks in the spline: (idx[k] - idx[k-1]) needs to be 900 seconds.
                spline['diff_idx'] = spline.index.to_series().diff().fillna(idx_delta)
                spline.loc[spline['diff_idx'] != idx_delta, 'diff_idx'] = 'Break'
                
                # Break the regular spline at the breaks (i.e., time is not continuous)
                splines_dict = ut.split_df_with(spline, 'diff_idx', split_key='Break', key_included= True, drop_cols=['diff_idx'])

                for key in splines_dict:
                    if key >= 0:                        # key -1 is the group for breakpoints
                        reg_spline = splines_dict[key]  # reg_spline: a dataframe that has regular continuous spline

                        if reg_spline.shape[0] >= min_datapoints:
                            reg_splines_list.append(reg_spline)
                            

        # Sort the splines on start time:
        reg_splines_list.sort(key= lambda x: x.iloc[0].name)

        print("\n-----------------------------------------------------\n")
        print("Done extracting regular glucose splines.")
        print("\n-----------------------------------------------------\n")
        
        if log == True:
            for i, s in enumerate(reg_splines_list):
                print("\n------------------")
                print("Regular spline ", i)
                print(s)
                    
        return reg_splines_list


    def _link_dfs_within_buffer(self, list_df, idx_buffer=(900,960), log=False):
        """
        args:
            list_df:    A list of sorted dataframes (sort by the start index of the dataframes).
            idx_buffer: (min, max) range of index buffer to merge dataframes whose end index is
                        within such index buffer with the start index of the next dataframes.
        return:
            merged_list_df: A new list of the merged dataframes.

        Given a list of sorted dataframes (sorted by their start index), and for each dataframe i,
        the idx_buffer offers a range of index buffer (i.e. (min, max)) to merge i with i+1 if 
        the end index of dataframe i and the start index of dataframe i+1 is within such range.
        The function will return a new list of the merged dataframes.
        """
        print("\n-----------------------------------------------------\n")
        print("Merging splines - If two splines are separated within the\n")
        print("time range: [{}, {}], they got merged.".format(idx_buffer[0], idx_buffer[1]))
        print("\n-----------------------------------------------------\n")

        min_range = idx_buffer[0]
        max_range = idx_buffer[1]

        merge_df = pd.concat(list_df)
        merge_df['diff_idx'] = merge_df.index.to_series().diff().fillna(idx_buffer[0])
        merge_df.loc[(merge_df['diff_idx']<min_range) ^ (merge_df['diff_idx']>max_range), 'diff_idx'] = 'Break'
        
        splines_dict = ut.split_df_with(merge_df, 'diff_idx', split_key='Break', key_included= True, drop_cols=['diff_idx'])
        
        new_splines = []
        for key in splines_dict:
            if key >= 0:                        # key -1 is the group for breakpoints
                reg_spline = splines_dict[key]
                new_splines.append(reg_spline)
        
        new_splines.sort(key= lambda x: x.iloc[0].name)

        print("\n-----------------------------------------------------\n")
        print("Done merging. The enhanced splines are:\n")
        print("\n-----------------------------------------------------\n")

        if log==True:
            for i, s in enumerate(new_splines):
                print("\n------------------")
                print("Enhanced spline ", i)
                print(s)

        return new_splines
    
    
    def plot_original_vs_cleaned_data(self, save=True):

        ######################################################       
        # Plot Figure 1: Data Cleaning and smooth
        ######################################################
        print("\n-----------------------------------------------------\n")
        print("Saving plots to the Log directory.")
        print("\n-----------------------------------------------------\n")
        output_dir = Path(self.log_dir)
        
        num_subplot = 3
        fig_width = int(self.total_time/86400) * 5   # 86400 sec is 1 day.
        fig_height = 2 * num_subplot
        axs_xlim = [0, self.total_time]
        axs_ylim = [2.5, 13]
        subplots_titles = [
            'Original Glucose',
            'Cleaned Glucose with Regular Splines Interval Extracted',
            'Gaussian Smoothed Glucose'
        ]

        fig, axs = ut.create_figure(fig_width, fig_height, num_subplot, axs_xlim, axs_ylim, subplots_titles)
        
        # Subfigure 1: Original glucose
        axs[0].plot(self.glucose_df)
        for i in range(int(self.total_time/86400)+1): # 86400 seconds = 24 hours
            axs[0].axvline(x=86400*i, ls='--', color='r', linewidth=0.5)
        axs[0].axhline(y=self.mean, ls='--', color='yellow', linewidth=0.5, label='mean')
        axs[0].axhline(y=self.mean + self.stdv, ls='--', color='orange', linewidth=0.5, label='mean + stdv')

        # Subfigure 2: Cleaned and extract data with 900 seconds intervals
        for n, frame in enumerate(self.regular_splines):
            axs[1].text(frame.index[0] + (frame.index[-1]-frame.index[0])/2, 8, n, fontsize=6)
            axs[1].plot(frame.index, frame['Glucose (mmol/L)'],'|-')
        axs[1].axhline(y=self.mean + self.stdv, ls='--', color='orange', linewidth=0.5, label='mean + stdv')
        axs[1].axhline(y=self.mean, ls='--', color='yellow', linewidth=0.5, label='mean')
        
        # Subfigure 3: Gaussian Smooth applied on subfigure 2 data.
        for frame in self.regular_splines:
            axs[2].plot(frame.index, gaussian_filter1d(frame['Glucose (mmol/L)'], 1), '|-')
        axs[2].axhline(y=self.mean + self.stdv, ls='--', color='orange', linewidth=0.5, label='mean + stdv')
        axs[2].axhline(y=self.mean, ls='--', color='yellow', linewidth=0.5, label='mean')
        
        # Save figure
        dc_plot_path = output_dir / '1_data_clean_{}.png'.format(self.subject_id)
        plt.savefig(dc_plot_path,format='png')
    

        ######################################################
        # Plot Figure 2: Utilize Data (Increase coverage)
        ######################################################
        num_subplot = 2
        subplots_titles = [
            'Cleaned Glucose (spline >= 1 hour, which is 4 datapoints)',
            'Cleaned Glucose allowed 1 mins buffer (between splines >= 1 hour)'
        ]
        
        fig, axs = ut.create_figure(fig_width, fig_height, num_subplot, axs_xlim, axs_ylim, subplots_titles)

        # Subfigure 1: Before the 1 minute buffer linkage 
        for n, frame in enumerate(self.regular_splines):
            axs[0].text(frame.index[0] + (frame.index[-1]-frame.index[0])/2, 8, n, fontsize=6)
            axs[0].plot(frame.index, frame['Glucose (mmol/L)'],'|-')
        axs[0].axhline(y=self.mean + self.stdv, ls='--', color='orange', linewidth=0.5, label='mean + stdv')
        axs[0].axhline(y=self.mean, ls='--', color='yellow', linewidth=0.5, label='mean')

        # Subfigure 2: After the 1 minute buffer linkage
        for n, frame in enumerate(self.merged_splines):
            axs[1].text(frame.index[0] + (frame.index[-1]-frame.index[0])/2, 8, n, fontsize=6)
            axs[1].plot(frame.index, frame['Glucose (mmol/L)'],'|-')
        axs[1].axhline(y=self.mean + self.stdv, ls='--', color='orange', linewidth=0.5, label='mean + stdv')
        axs[1].axhline(y=self.mean, ls='--', color='yellow', linewidth=0.5, label='mean')

        if save == True:
            # Save figure
            linkage_plot_path = output_dir / '2_data_enhance_{}.png'.format(self.subject_id)
            plt.savefig(linkage_plot_path,format='png')
            plt.close('all')
        else:
            plt.show()

    def _define_endpoint_range(self, baseline_mode):
        """    
        Baseline mode is a pre-defined instruction to define how the peak's baseline range should be computed.
        For example, baseline mode '1' is defined as [mean, mean + 0.5 stdv], then this function is to return
        the numeric range of the left and right endpoint of the peak. It returns:

            1. l_endpoint_range
            2. r_endpoint_range
            3. baselines_description
        
        One can add more baseline pattern as required, and note that it is possible to allow different range
        for left and right endpoints of the peak, respectively. The 'extract_peaks()' function can
        handle asymmetric endpoint baselines for a peak, i.e., left and right endpoints fall in different range.   
        """
        baselines_descrip = {
            0: 'None, None',
            1: '[mean, mean + 0.5 stdv]',
            2: '[mean + 0.5 stdv, mean + stdv]',
            3: '[mean - 0.5 stdv, mean]',
            4: '[mean - stdv, mean - 0.5 stdv]',
            5: '[mean, mean + stdv]',
            6: '[mean - stdv, mean]',
            7: '[mean - 0.5 stdv, mean + 0.5 stdv]'
        }

        l_endpoint_range = None
        r_endpoint_range = None

        mean = self.mean
        stdv = self.stdv

        if baseline_mode == 0:
            pass
        if baseline_mode == 1:
            l_endpoint_range = [mean, mean + 0.5*stdv]
            r_endpoint_range = [mean, mean + 0.5*stdv]
        elif baseline_mode == 2:
            l_endpoint_range = [mean + 0.5*stdv, mean + stdv]
            r_endpoint_range = [mean + 0.5*stdv, mean + stdv]
        elif baseline_mode == 3:
            l_endpoint_range = [mean - 0.5*stdv, mean]
            r_endpoint_range = [mean - 0.5*stdv, mean]
        elif baseline_mode == 4:
            l_endpoint_range = [mean - stdv, mean - 0.5*stdv]
            r_endpoint_range = [mean - stdv, mean - 0.5*stdv]
        elif baseline_mode == 5:
            l_endpoint_range = [mean, mean + stdv]
            r_endpoint_range = [mean, mean + stdv]
        elif baseline_mode == 6:
            l_endpoint_range = [mean - stdv, mean]
            r_endpoint_range = [mean - stdv, mean]
        elif baseline_mode == 7:
            l_endpoint_range = [mean - 0.5 * stdv, mean + 0.5 * stdv]
            r_endpoint_range = [mean - 0.5 * stdv, mean + 0.5 * stdv]

        return l_endpoint_range, r_endpoint_range, baselines_descrip

    def clean_off_peaks(self):
        """
        Reset the peaks dict to empty dict.
        """
        self.peaks_dict = {}


    def extract_peaks(self, peak_min=None, baseline_mode=1, gaussian_stdv=1):
        """
        This function is the main function to extract peaks which satisfies the baseline configuration 
        in the continuous glucose data. The purpose of baseline configuration is to rule out those peaks which
        have too low endpoints (ie., usually small peaks) or too high (i.e., usually are early-stopping
        incomplete peaks). Each time this function is run, it adds an entry to the class attribute 'self.peaks_dict', 
        with the 'baseline mode' as key, and has a dict as its value:
        {'peaks' : peaks, 'number_of_peaks' : len(peaks), 'Gaussian_smooth' : gaussian_stdv} 
        
        Note: For more details about the baseline modes, refer to '_define_endpoint_range()' 

        args:
            - peak_min:     (1) None: Not specified.
                            (2) positive float or int: 
                                The minimum threshold of the top point for a peak to be kept.
            - baseline_mode: for baseline integer defined in '_define_endpoint_range()' function.
        
        return:       
            - list of peaks: a list of dataframes, where each of them is the peak found.

        """

        l_endpoint_range, r_endpoint_range, baseline_descrip  = self._define_endpoint_range(baseline_mode)
        
        peaks = []
        for frame in self.regular_splines:
            peaks += fp.find_peak_pattern(frame, 'Glucose (mmol/L)', peak_min=peak_min, gaussian_stdv=gaussian_stdv, start_range=l_endpoint_range, end_range=r_endpoint_range)

        # Save the extracted peaks and parameters to the class attribute
        self.peaks_dict[baseline_mode] =   {'baseline_descrip' : baseline_descrip[baseline_mode],
                                            'peaks' : peaks, 
                                            'number_of_peaks' : len(peaks), 
                                            'Gaussian_smooth' : gaussian_stdv }

        return {baseline_mode: self.peaks_dict[baseline_mode]}


    def plot_baseline_peaks(self, baselines = [], save=True):
        """
        This function will plot 2 subplots for the pre-filtered peaks: 
            1/ Gaussian Smoothed Glucose.
            2/ Peaks from original glucose, using Gaussian peaks as mask.

        and N many number of subplots for the filtered peaks, based on different baselines.
            For the peak data stored in class attribute 'self.peaks_dict', specifies which
            baseline modes to plot in argument 'baselines'. Add baseline_modes to
            to be shown in subplots.

                
        This function generates Figure 3 in the log_dir: 
        '1/ Gaussian Smoothed Glucose: This plot reconstruct how the smoothed
            glucose looks like using the specified Gaussian standard deviation value.
        '2/ Peaks from original glucose: This plot shows the peaks extracted
            from the original glucose using the Gaussian smooth values as mask.
        '3/ **Filterd_peak_with_baseline_mode_{x}**: This plot is the final result
            of the peaks extraction, with filters of the baseline_mode {x}. 
        """
        print("\n-----------------------------------------------------\n")
        print("Saving plots to the Log directory.")
        print("\n-----------------------------------------------------\n")

        keys_in_peak_dict = baselines

        if not self.peaks_dict:
            print('No peaks to plot. Please extract peaks first.')
            return
        elif not len(keys_in_peak_dict):
            keys_in_peak_dict = sorted(list(self.peaks_dict.keys()))
        elif not all(key in self.peaks_dict for key in keys_in_peak_dict):
            print('One or more keys not found in peak_dict. Please double check the keys.')
            return

        # Baseline 0 is identical to subplot 2/ No need to plot again.
        if 0 in keys_in_peak_dict: keys_in_peak_dict.remove(0)
        ###########################################################
        # Figure 3: Peak Extraction
        ###########################################################
        num_subplot = len(keys_in_peak_dict) + 2
        fig_width = int(self.total_time/86400)*5
        fig_height = 2 * num_subplot
        axs_xlim = [0, self.total_time]
        axs_ylim = [2.5,13]
        subplots_titles = [
            '1/ Gaussian Smoothed Glucose',
            '2/ Peaks from original glucose',
        ]
        for baseline in keys_in_peak_dict:
            subplots_titles.append('3/ Filterd_peak_with_baseline_mode_{}_{}'.format(baseline, self.peaks_dict[baseline]['baseline_descrip']))
   

        fig, axs = ut.create_figure(fig_width, fig_height, num_subplot, axs_xlim, axs_ylim, subplots_titles)

        # Gaussian_smooth is the standard deviation used to extract the peaks.
        gaussian_stdv = self.peaks_dict[keys_in_peak_dict[0]]['Gaussian_smooth']

        # Subfigure 1: Gaussian Smoothed Glucose
        for frame in self.regular_splines:
            axs[0].plot(frame.index, gaussian_filter1d(frame['Glucose (mmol/L)'], gaussian_stdv), '|-')
        axs[0].axhline(y=self.mean + self.stdv, ls='--', color='orange', linewidth=0.5, label='mean + stdv')
        axs[0].axhline(y=self.mean, ls='--', color='yellow', linewidth=0.5, label='mean')
        

        # Subfigure 2: Extract peaks in the orginal glucose data by the Gaussian smooth mask.
        baseline = 0   # No baseline configured (Peaks can fall in any baseline range.) 
        l_endpoint_range, r_endpoint_range, baseline_descrip = self._define_endpoint_range(baseline)
        
        general_peaks = []
        for frame in self.regular_splines:
                general_peaks += fp.find_peak_pattern(frame, 'Glucose (mmol/L)', peak_min=None, gaussian_stdv=gaussian_stdv, start_range=l_endpoint_range, end_range=r_endpoint_range)

        for n, p in enumerate(general_peaks):
            axs[1].text(p.index[0] + (p.index[-1]-p.index[0])/2, 8, n, fontsize=6)
            axs[1].plot(p.index, p['Glucose (mmol/L)'], '|-')
    
        axs[1].axhline(y=self.mean + self.stdv, ls='--', color='orange', linewidth=0.5, label='mean + stdv')
        axs[1].axhline(y=self.mean, ls='--', color='yellow', linewidth=0.5, label='mean')    
        
        # Subfigure 3 - N*: Filter peaks in subfigure 2 with baseline range.
        for i in keys_in_peak_dict:
            
            baseline = i
            peak_dict = self.peaks_dict[baseline]
            baseline_descrip = peak_dict['baseline_descrip']
            peaks = peak_dict['peaks']
            l_endpoint_range, r_endpoint_range, baseline_descrip = self._define_endpoint_range(baseline)

            ax_index = 1 + i

            for n, p in enumerate(peaks):
                axs[ax_index].text(p.index[0] + (p.index[-1]-p.index[0])/2, 8, n, fontsize=6)
                axs[ax_index].plot(p.index, p['Glucose (mmol/L)'], '|-')
                
            if l_endpoint_range != None:
                axs[ax_index].axhline(y=l_endpoint_range[0], ls='--', color='red', linewidth=0.5, label='L-endpoint range')
                axs[ax_index].axhline(y=l_endpoint_range[1], ls='--', color='red', linewidth=0.5, label='L-endpoint range')
            if r_endpoint_range != None:
                axs[ax_index].axhline(y=r_endpoint_range[0], ls='--', color='green', linewidth=0.5, label='R-endpoint range')
                axs[ax_index].axhline(y=r_endpoint_range[1], ls='--', color='green', linewidth=0.5, label='R-endpoint range')     
    
        if save==True:
            peaks_extraction_plot_path = self.log_dir / '3_peaks_baseline_{}_{}.png'.format(baseline, self.subject_id)
            plt.savefig(peaks_extraction_plot_path, format='png')
            plt.close('all')
        else:
            plt.show()

        print("\n-----------------------------------------------------\n")
        print("Done saving plots")
        print("\n-----------------------------------------------------\n")


    def extract_peaks_on_baselines(self, keys_in_peak_dict=[0,1,2,3,4,5,6,7], gaussian_stdv=1):
        """
        This function would run 'extract_peaks()' on all the baseline modes specified in the argument.
        Default is running all 7 baseline modes. 
        """
        # Clean up the pre-existing extraction.
        self.peaks_dict = {}

        baseline_modes = keys_in_peak_dict

        for baseline in baseline_modes:        
            self.extract_peaks(peak_min=self.mean+self.stdv, baseline_mode=baseline, gaussian_stdv=gaussian_stdv)

        return self.peaks_dict


    def top_N_peaks_extraction(self, baselines=[7,6,5], top_N=5):
        """
        This function picks the top N peaks from a set of extracted peaks, possibly
        from different baselines set. 
        
        [Logic]
        Baseline 0 is the general peaks: without any restriction on the peak's endpoints
        on both side. However, these peaks can be imbalanced (i.e., two legs are not
        sitting in the same ground). General peaks are used to build priority_peaks,
        a list of peaks sorted by their maximum value. Then, given this priority_peaks,
        from the first top peak, it checks whether the peak is within the baseline range of
        interest. If it is found in the specified baselines range, then it is added to the
        'self.top_peaks'. The idea of imposing the baselines restriction is to find
        complete peaks, and filter out those highly inbalanced ones. 

        args:
            - key_in_peak_dict:  the baseline_mode key.
            - top_N:             the number of top peaks, sorted by their maximum values.
        """

        
        if any(b not in self.peaks_dict for b in baselines):
            raise Exception('One or more keys are not found in self.Peaks_dict. Please recheck the keys.')

        if 0 in self.peaks_dict:
            general_peaks = self.peaks_dict[0]['peaks']
        else:
            general_peaks = self.extract_peaks(baseline_mode=0)

        col = 'Glucose (mmol/L)'
        priority_peaks = sorted(general_peaks, key=lambda x: x[col].max(), reverse=True)

        # Sorting from the widest baseline range to the narrowest. From empirical results, baseline 7
        # and baseline 6 captures most of the balanced peaks, with the wider tolerance on the range
        # where the endpoint legs need to fall in.
        baselines.sort(reverse=True)
        
        for p in priority_peaks:
            for b in baselines:
                baseline_peaks = self.peaks_dict[b]['peaks']
                balanced_p = None
                for b in baseline_peaks:
                    if b.index[0] == p.index[0]:
                        balanced_p = b
                        break
                if isinstance(balanced_p, pd.DataFrame):
                    self.top_peaks.append(balanced_p)
                    break
            if len(self.top_peaks) >= top_N:
                break
        
        return self.top_peaks
        
    def plot_top_peaks_vs_manual_peaks(self, save=True):
        
        print("\n-----------------------------------------------------\n")
        print("Saving plots to the Log directory.")
        print("\n-----------------------------------------------------\n")
        
        num_fig = 3
        fig_width = int(self.total_time/86400)*5
        fig_height = 2 * num_fig
        axs_xlim = [0,self.total_time]
        axs_ylim = [2.5,13]
        fig_titles = ['1/ General Peaks from the original glucose data',
                      '2/ Manually selected Peaks from the original glucose data',
                      '3/ Automatically selected Peaks from the original glucose data']

        fig,axs = ut.create_figure(fig_width, fig_height, num_fig, axs_xlim, axs_ylim, fig_titles)

        # Subfigure 1: General peak patterns extracted from the original glucose.
        baseline = 0
        
        if baseline in self.peaks_dict:
            general_peaks = self.peaks_dict[baseline]['peaks']
        else:
            general_peaks = self.extract_peaks(baseline_mode=0)


        for n, p in enumerate(general_peaks):
            axs[0].text(p.index[0] + (p.index[-1]-p.index[0])/2, 8, '{}:{}'.format(n, len(p.index)-1), fontsize=6)
            axs[0].plot(p.index, p['Glucose (mmol/L)'], '|-')

        axs[0].axhline(y=self.mean+self.stdv, ls='--', color='green', linewidth=0.5, label='mean + stdv')
        axs[0].axhline(y=self.mean, ls='--', color='red', linewidth=0.5, label='mean')
        axs[0].axhline(y=self.mean-self.stdv, ls='--', color='green', linewidth=0.5, label='mean - stdv')
        axs[0].axhline(y=7.8, ls='--', color='blue', linewidth=0.5, label='high bound')
        

        # Subfigure 2: Plot manual selected peaks
        if len(self.manual_peaks):
            for n, p in enumerate(self.manual_peaks):
                axs[1].text(p.index[0] + (p.index[-1]-p.index[0])/2, 8, '{}:{}'.format(n, len(p.index)-1), fontsize=6)
                axs[1].plot(p.index, p['Glucose (mmol/L)'], '|-')

        axs[1].axhline(y=self.mean+self.stdv, ls='--', color='green', linewidth=0.5, label='mean + stdv')
        axs[1].axhline(y=self.mean, ls='--', color='red', linewidth=0.5, label='mean')
        axs[1].axhline(y=self.mean-self.stdv, ls='--', color='green', linewidth=0.5, label='mean - stdv')
        axs[1].axhline(y=7.8, ls='--', color='blue', linewidth=0.5, label='high bound')
        
    
        #### Plot Top 5 auto selected peaks
        for n, p in enumerate(self.top_peaks):
            axs[2].text(p.index[0] + (p.index[-1]-p.index[0])/2, 8, '{}:{}'.format(n, len(p.index)-1), fontsize=6)
            axs[2].plot(p.index, p['Glucose (mmol/L)'], '|-')

        axs[2].axhline(y=self.mean+self.stdv, ls='--', color='green', linewidth=0.5, label='mean + stdv')
        axs[2].axhline(y=self.mean, ls='--', color='red', linewidth=0.5, label='mean')
        axs[2].axhline(y=self.mean-self.stdv, ls='--', color='green', linewidth=0.5, label='mean - stdv')
        axs[2].axhline(y=7.8, ls='--', color='blue', linewidth=0.5, label='high bound')
        
        if save == True:
            # Save plots
            peaks_filter_plot_path = self.log_dir / '5_machine_vs_human_peaks_{}.png'.format(self.subject_id)
            plt.savefig(peaks_filter_plot_path,format='png')
            plt.close('all')
        else:
            plt.show()

    def _manual_peaks(self):
        """
        This function is for ad-hoc use to extract the previously labeled human selected peaks in the CSV files.
        The manual peaks are under the columns which name is prefixed with 'peak'.
        """

        df = pd.read_csv(self.csv_path)
        df['Elapsed Time'] = df['Elapsed Time'].astype(int)
        df = df.set_index('Elapsed Time')
        df['peaks'] = 0

        columns = [i for i in df.columns if i.startswith('peak')]
        
        for c in columns:
            peak_pattern = df.loc[:, c].dropna().to_numpy()
            df['peaks'] += ut.find_pattern(df, peak_pattern, 'Glucose (mmol/L)', backfill=True)
        
        df = df.loc[:, ['Glucose (mmol/L)', 'peaks']]
        peaks = ut.split_df_with(df, 'peaks', split_key=0, key_included=False, drop_cols=['peaks'])
        manual_selection_peaks = [peaks[i] for i in peaks if i != -1]

        # If no manual peaks found in the file, it's empty list.
        return manual_selection_peaks

def run_batch_and_save_peaks_to_yaml(list_glucoseCSV, glucose_dir, peak_yaml_filename, peak_yaml_dir, log_output_dir):
    
    # The main dict that will be saved to peak yaml file
    batch_peak_data = []

    for glucoseCSV in list_glucoseCSV:
        print("===========================================================")
        print("Start processing:", glucoseCSV)
        print("===========================================================")
        csv_file = glucose_dir / glucoseCSV
        
        subj = PeakSelection(csv_file, log_output_dir, data_cleanse_plot=True)

        # Each subject has a personal dict that will be added to batch_peak_data
        subject_data = {}
        subject_data['subject'] = subj.subject_id

        # Add stats
        subject_data['avg'] = float(format(subj.mean, '.9f'))
        subject_data['stdv'] = float(format(subj.stdv, '.9f'))

        # Add Manually selected glucose peaks reading from csv, if any.
        if len(subj.manual_peaks):
            manual_selection_peaks = [p['Glucose (mmol/L)'].to_numpy() for p in subj.manual_peaks]
        else:
            manual_selection_peaks = []
        
        # for i, p in enumerate(manual_selection_peaks):
        #     print("manual peak", i, ": ")
        #     print(p)
        #     print()

        if (len(manual_selection_peaks) == 0):
            print("The file {} does not have manual peaks records.".format(glucoseCSV))
        else:
            if (not manual_selection_peaks[0].shape[0] == 0):
                
                check_len = len(manual_selection_peaks[0])
                if all(len(i)==check_len  for i in manual_selection_peaks):
                    manual_avg_peak = sum(manual_selection_peaks)/len(manual_selection_peaks)
                    manual_max_val = max(map(max, manual_selection_peaks))
                    manual_norm_peak = manual_avg_peak / manual_max_val

                    subject_data['g_original'] = {}
                    subject_data['g_original']['values'] = [float(format(i, '.9f')) for i in manual_norm_peak.tolist()]
                    subject_data['g_original']['max_val'] = float(format(manual_max_val, '.9f'))
                else:
                    print("manual peaks have different length, can't perform average.")
            

        # Add top 5 machine selected peaks
        subj.extract_peaks_on_baselines(keys_in_peak_dict=[0,1,2,3,4,5,6,7], gaussian_stdv=1)
        subj.plot_baseline_peaks(save=True)
        subj.top_N_peaks_extraction(baselines=[7,6,5], top_N=5)
        subj.plot_top_peaks_vs_manual_peaks(save=True)
        
        for i, p in enumerate(subj.top_peaks):
            if len(subj.top_peaks) != 0:
                peak_id = 'peak_' + str(i)
                glucose = p['Glucose (mmol/L)'].values.tolist()
                max_val = max(glucose)
                norm_glucose = [i/max_val for i in glucose]

                subject_data[peak_id] = {}
                subject_data[peak_id]['max_val'] = float(format(max_val, '.9f'))
                subject_data[peak_id]['values'] = [float(format(i, '.9f')) for i in norm_glucose]

                print("top peak", i, ": ")
                print(p)
                print()

        # Add subjecct data to batch
        batch_peak_data.append(subject_data)

        print("===subject data dict===================")
        print(subject_data)

    output_peak_dict = {'SUBJECTS': {'DATA' : batch_peak_data}}
    output_yaml = peak_yaml_dir/ peak_yaml_filename
    
    with open(output_yaml, 'w') as yaml_file:
        yaml.dump(output_peak_dict, yaml_file, default_flow_style=False, sort_keys=False)
        print(output_yaml)

def batch_PS():
    import argparse

    from src.config import cfg
    ######################################################
    # Config Path
    ######################################################
    # Take experiment configuration
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

    # Setup Log dir for the batch experiment
    output_dir = Path(cfg.LOG_DIR)
    if not output_dir.exists():
            print('=> creating {}'.format(output_dir))
            os.makedirs(output_dir)
            # output_dir.mkdir()

    if not peak_dir.exists():
        os.makedirs(peak_dir)
    

    # Setup sub dir for each experiment config 
    cfg_name = os.path.basename(args.cfg).split('.')[0]
    time_str = time.strftime('%Y_%m_%d_%H_%M')
    final_output_dir = output_dir / cfg_name / (cfg_name + '_peak_' + time_str)
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    ######################################################       
    # Run batch peakSelection
    ######################################################
    glucose_dir = Path(cfg.GLUCOSE_DATA_DIR)

    filenames = os.listdir(glucose_dir)
    csv_files = [ i for i in filenames if i.endswith('csv') ]
    
    # Run all subjects    
    run_batch_and_save_peaks_to_yaml(csv_files, glucose_dir, peak_yaml_filename, peak_dir, final_output_dir)



if __name__ == "__main__":
    
    import argparse
    from config import cfg

    ######################################################
    # Config Path
    ######################################################
    # Take YAML configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)


    args = parser.parse_args()
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # Set up peak yaml foler to save extracted peaks
    peak_dir = Path(cfg.PEAK_DATA_DIR)
    peak_yaml_filename = Path(cfg.PEAK_DATA_FILE)

    # Setup Log dir for the batch experiment
    output_dir = Path(cfg.LOG_DIR)
    if not output_dir.exists():
            print('=> creating {}'.format(output_dir))
            output_dir.mkdir()
    

    # Setup sub dir for each experiment config 
    cfg_name = os.path.basename(args.cfg).split('.')[0]
    time_str = time.strftime('%Y_%m_%d_%H_%M')
    final_output_dir = output_dir / cfg_name / (cfg_name + '_peak_' + time_str)
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    
    ######################################################       
    # Tutorial:  Read subjects' glucose data
    ######################################################
    glucose_dir = Path(cfg.GLUCOSE_DATA_DIR)
    
    filenames = os.listdir(glucose_dir)

    # Specify a CSV file
    csv_file = glucose_dir / "0M0008AQUF4.csv"

    # Initialize class instance: data cleanse and basic statistics will be built
    # If plot==True, the visualization of data cleanse process will be saved.
    subj = PeakSelection(csv_file, final_output_dir, data_cleanse_plot=True) 

    # Print the stats:
    print("subject id:", subj.subject_id)
    print("total length of glucose:", subj.total_time)
    print("mean glucose:", subj.mean)
    print("standard deviation:", subj.stdv)
    print("manual peaks found in the file:", bool(len(subj.manual_peaks)))

    # Start extract the peaks set on baseline 0: 
    # Baseline 0 is the general peaks, and is free ranged. 
    #   It extracts all the peaks found in the original glucose data.
    subj.extract_peaks(baseline_mode=0, gaussian_stdv=1)
    print("peaks extracted:")
    pprint.pprint(subj.peaks_dict[0])
    print()

    # Try extract the peaks set on baseline 1:
    # Check '_define_endpoint_range' method for more baseline modes configuration.
    subj.extract_peaks(baseline_mode=1, gaussian_stdv=1)
    print("peaks extracted:")
    pprint.pprint(subj.peaks_dict[1])
    print()

    # Try extract another peaks set on baseline 2:
    subj.extract_peaks(baseline_mode=2, gaussian_stdv=1)
    print("peaks extracted:")
    pprint.pprint(subj.peaks_dict[2])
    print()

    # Extract all baselines peaks at once
    subj.extract_peaks_on_baselines(keys_in_peak_dict=[0,1,2,3,4,5,6,7], gaussian_stdv=1)
    print("All baseline extracted:")
    pprint.pprint(subj.peaks_dict)
    print()

    # Plot it to visualize it. Use save=False to real-time show on screen.
    subj.plot_baseline_peaks(save=True)
    
    # Extract top 5 peaks from baselines.
    subj.top_N_peaks_extraction(baselines=[7,6,5], top_N=5)
    subj.plot_top_peaks_vs_manual_peaks(save=True)