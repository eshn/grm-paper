import pandas as pd
import numpy as np
import src.utils.utils as ut
from scipy.ndimage.filters import gaussian_filter1d



def crop_peak(df, col_name, peak_min = None, start_range = None, end_range = None, lcrop_step = 0, rcrop_step = 0):
    """
    args: 
        df: the dataframe of one peak pattern.
        col_name: the column which the values reside.
        peak_min:   (1) None: Not specified.
                    (2) positive float or int: the minimum threshold of the top point for a peak to be kept.
        start_range: (1) None: No boundary (2) [lower bound, upper bound] 
        end_range:  (1) None: No boundary (2) [lower bound, upper bound]
        lcrop_step: (1) -1: no cropping performs on the left endpoint.
                    (2) positive int: maximum steps can be cropped from the left most.
                    (3) 0: cropped from the left most without steps constraint.
        rcrop_step: (1) -1: no cropping performs on the right endpoint.
                    (2) positive int: maximum steps can be cropped from the right most.
                    (3) 0: cropped from the right most without steps constraint.

    return:
        df: the filtered (or cropped) dataframe of the peak pattern.

    The function takes a dataframe of a peak pattern, and found boundaries from the
    beginning and ending points. If no crop_step specified, it only checked whether the 
    end points meet the start_range and end_range. If it satisfies, it would return the
    dataframe; otherwise, it returns an empty dataframe. 
    
    If the l_crop_step (left cropping steps) or r_crop_step (right cropping steps) is set,
    the function would search from the leftmost or/and rightmost data points, until it found
    a new start or/and end point is within range. It returns the new cropped dataframe if found;
    otherwise, it returns an empty dataframe.
    """

    df = df.copy()
    

    if (start_range != None or end_range != None):                    
        if start_range != None and end_range != None:
            baseline = [min(start_range), min(end_range)]
        elif start_range == None:
            baseline = [min(end_range)]
        else:
            baseline = [min(start_range)]
    
        # Check peaks' maximum threshold: if not qualified, drop it
        if peak_min == None:
            if df[col_name].max() <= min(baseline):
                df = df[0:0]
        elif peak_min >=0:
            if df[col_name].max() < peak_min:
                df = df[0:0]

        # The peak's maximun is higher than the baseline: find the left and right boundary 
        if not df.empty:
            if (lcrop_step >=0) or (rcrop_step >= 0):
                peak_iloc = np.argmax(df[col_name])
                end_iloc = len(df[col_name]) - 1
                left_bound = None
                right_bound = None
                
                if start_range == None:
                    left_bound = 0
                else:
                    # Set the left search end index:
                    if int(lcrop_step) == 0:
                        l_search = peak_iloc
                    elif lcrop_step > 0:
                        l_search = min(lcrop_step, peak_iloc)
                        
                    # Start searching left bound in dataframe.
                    for i in range (0, l_search):
                        left_point = df[col_name].iloc[i]
                        if (left_point >= start_range[0]) and (left_point <= start_range[1]):
                            left_bound = i
                            break
                    
                if end_range == None:
                    right_bound = end_iloc
                else:
                    # Set the right search end index:
                    if int(rcrop_step) == 0:
                        r_search = peak_iloc
                    if rcrop_step > 0:
                        r_search = max(end_iloc-rcrop_step, peak_iloc)

                    # Start searching right bound in dataframe.
                    for j in range (end_iloc, r_search, -1):
                        right_point = df[col_name].iloc[j]
                        if (right_point >= end_range[0]) and (right_point <= end_range[1]):
                            right_bound = j
                            break

                # Find no boundaries matches the baseline: drop it
                if (left_bound == None or right_bound == None):
                    df = df[0:0]
                # Find boundaries matched: crop the peak pattern
                else:
                    df = df.iloc[left_bound:right_bound+1]
            
            # Default: simply filter those peaks whose handles are within ranges.
            else:
                left_point = df[col_name].iloc[0]
                right_point = df[col_name].iloc[-1]

                # If the peak's endpoints are not within the boundary ranges: drop it
                if not ((left_point >= start_range[0]) and (left_point <= start_range[1]) and \
                        (right_point >= end_range[0]) and (right_point <= end_range[1])):
                    df = df[0:0]

    return df


def find_peak_pattern(df, col_name, peak_min=None, gaussian_stdv=1, start_range = None, end_range = None):
    """
    args:
        df:           The target dataframe
        col_name:     The column from which the peaks are to extract
        gaussian_std: The standard deviation to be used in Gaussian mask.
        drop:         If True, all the values below this level would be cropped.
        baseline:     Define the baseline above which the peaks to be found.
                      Note this value only matters is drop=True.
        start_range:  A 2-element list or tuple, i.e., [low, high], or None.
                      [low, high]     - the range allowed for the left-most endpoint.
                      [-np.inf, high] - no lower bound.
                      [low, np.inf]   - no upper bound
                      None            - allow all range
        end_range:    Same as start_range.

    return:
        parabola_list: A list of dataframes, each of which is a peak pattern.

    This function takes a dataframe and applies gaussian smoothing on the column values
    to remove the noise. Use the smooth version to identify the start and end point of
    the peak patterns from the original glucose data. The returned value is a break-down
    list of the dataframes of each peak.

    ================================================================================
    * Logic of this function:
    1. 1st derievative: Take a diff() on glucose data.
        1: if the glucose is increasing
        0: if the glucose is declining or no change.
    
    2. 2nd derievative: Take a diff() again.
        1: neg -> pos      (1-0=1)         -> The two end points of the peak
        0: pos -> neg      (max(0-1, 0))   -> The turning point of the peak
        0: no change (pos->pos; neg->neg)  -> Points in the middle of the curves
    
    3. The goal is to crop the section of a peak, identified by [0,1] boundaries, 
       which suggests the slope is turning from negative/no-change to positive.
        e.g., given a glucose sequence:
              5 4 6 8 7 4 5 6 7 5 2 1 2
              the sequence of 1st deriv.:
              0 0 1 1 1 0 1 1 1 0 0 0 1
              the sequence of 2nd deriv.:
              0 0 1 0 0 0 1 0 0 0 0 0 1
              find_pattern([0, 1]).shift(-1).cumsum(): Assign group to each peak
              0 1 1 1 1 2 2 2 2 2 2 3 3
    
        Note: The end point of a peak can be simultaneously the start point of the
              next peak. In the peak grouping step, such point (the overlapping end
              point and start point of the next peak) is labeled as the starting
              point of the next peak. Thus, an appending of the end-point would be
              applied to include the last point in the peak.
              Exception: when it's the final index, where no next index available.

    4. Deal with the last segment:
       The last group is not bound with two endpoints (start/ end of the peak), i.e., 
       bounded with two [0, 1] boundaries. It is possible that the last section
       is not a complete peak, e.g., when the glucose is only climbing. Thus,
       do a simple comparison of the last two data points: If it's climbing, drop the
       group; If it's declining, keep it.
    """

    df = df.copy()

    df['gaussian'] = gaussian_filter1d(df[col_name], gaussian_stdv)

    # 1. First order derivatives:
    df['deriev'] = df['gaussian'].diff().fillna(0)
    df.loc[df['deriev']>0, 'deriev'] = 1
    df.loc[df['deriev']<0, 'deriev'] = 0

    # 2. Second order derivatives:
    df['deriev'] = df['deriev'].diff().fillna(0)
    # Identify the turning point of a curve: changing from negative slope to positive slope (i.e., the V shape).
    df['deriev'] = ut.find_pattern(df, [0, 1], 'deriev', backfill=False).shift(-1).fillna(0.0).cumsum()
    
    # The last segment group: Group 0 is placeholder to drop
    if df.iloc[-1]['gaussian'] >= df.iloc[-2]['gaussian']:
        df.loc[df['deriev']==df.iloc[-1]['deriev'], 'deriev'] = 0
    
    parabola_list = []
    last_idx = df.iloc[-1].name

    
    for g, rows in df.groupby('deriev').groups.items():
        # Minimum points that form a peak: 3
        if (len(rows)>=2) and (int(g)!=0):
            
            # Extract Peak pattern
            if rows[-1] != last_idx:
                new_df = df.loc[rows].copy().append(df.loc[df.index[df.index.get_loc(rows[-1])+1]])
            else:
                new_df = df.loc[rows].copy()

            # Extract Peaks that match baseline requirement on the left and right boundaries.
            new_df = crop_peak(new_df, col_name, peak_min=peak_min, start_range= start_range, end_range= end_range, lcrop_step = 0, rcrop_step = 0)

            if not new_df.empty:
                if new_df.shape[0] >=3 :
                    new_df.drop(columns=['gaussian', 'deriev'], inplace=True)
                    parabola_list.append(new_df)
    
    return parabola_list