# Top level code for the gradient descent. 
# Mostly by Jacob Morra, tweaked and adjusted by Lennaert van Veen, 2019.
import yaml
import argparse
from copy import copy
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from operator import itemgetter
import os
from pathlib import Path
import pprint
import sys
from scipy.interpolate import CubicSpline
import time
# The following are all bots of our own code:
from u_calc import u_calc # Compute the control variable u for given (history of) e (excess glucose level).
from e_j import e_j       # Short function that prepends zeros before the known time series of e.
from e_calc import e_calc # Compute time series for e for given time series of u.
from peak import peak     # Compute the infut peak F(t) assuming a Gaussian shape.
from sse import sse       # Compute the distance between the measured and model time series of e (can probably be replaced by a single call to a numpy function!).
from Ecalc import Ecalc   # Function that computes the cost function E.
from e_get import e_get   # Function that first computes u and then e.
from gradDesc import gradients # Finite difference routines to approximate the gradient of E.
from e_interp import e_interp               # Uses cubic spline interpolation to determine values of e in between grid points (may not be used!).
from u_interp import u_interp               # Same for u.
from convertGtoE import convertGtoE         # Converts the "representative peak" to the peak e by subtracting the set point and, optionally, adding zeros.
from createInputPeak import createInputPeak # Returns the values of F(t).
from read_peaks import read_peak            # Returns the subject's peak data from peak.yaml
from src.config import cfg
import src.utils.plotRadarFactory as radarplot

class subjectFitToModel:

        def __init__(self, subj_id, peak_dict, log_dir):    

                self.subject_id = subj_id
                self.peaks = peak_dict
                self.log_dir = log_dir
                
                self.avg = peak_dict['avg']
                self.stdv = peak_dict['stdv']
                
                self.peaks = {}
                if 'g_original' in peak_dict:
                        self.peaks['g_original'] = peak_dict['g_original']
                auto_peaks_keys = [k for k in peak_dict if k.startswith('peak_')]
                for k in auto_peaks_keys:
                        self.peaks[k] = peak_dict[k]

                self.params = {}
                self.params_log = {}
                

        # Comment: adding zeros to the end of the peak helps. Without them the control variable does not go back to zero and it seems the gradient descent cannot converge. However, if you add too many zeros the time-stepping for computing e becomes unstable and the whole thing blows up. This number seems to be just right.
        # Set the base line to the minimum over the representative peak and subtract it from the glucose data; add zeros at the end.
        def gradientDescent(self, glucose):
                """
                @glucose:  an array of normalized glucose values 
                """

                avg = np.min(glucose)
                e_original = convertGtoE(glucose,avg,2)

                # Initialize parameters ----------------------------------------------------------------------------------------------
                
                zeros = [0]*len(e_original)
                tstep = 1.0 # Time step between measurements (normalized to 1, i.e. we measure in units of 15 mins)
                # Reasonable starting point for the parameters.
                lam = 0.4116421339794699
                A1 = 0.1
                A2 = 0.1
                A3 = 0.005      # Fixes base metabolic rate.
                A4 = 1.0        # Fixed to unity by nondimensionalization.
                h = 1.0         # Time step for explicit Euler approximation of e (in units 15 mins)
                # The height (ampl) should be about equal to the maximum of e_original, the shift and the width should be such that the "meal" takes about 15-45 minutes and ends around the time e_original peaks
                sig = 1.0 # Standard deviation in F
                a = sig
                # Try setting the shift automatically based on the max of the input data...
                peak_loc = np.argmax(e_original)
                shift = float(peak_loc) - 1.0   # Setting the shift so that the peak of the meal is 15m ahead of the glucose peak.
                amp = 1.7524890366951094        # Amplitude for the input peak.
                e0 = e_original[0]              # Initial point for the time integration of e.
                numSteps = len(e_original)      # Length of the time series to produce from the model.
                delta = 0.01                    # Finite difference parameter. This is now a fraction, i.e. the actual finite difference parameter will be (delta * A) for the derivative wrt A. This is important because the parameters may become as small as delta during the gradient descent iteration, in which case the approximation of the derivative would break down.
                sc = 0.0001                     # Scaling of the gradient for the descent step. Comment: this is now fixed, but we may have to change it depending on the convergence speed.
                sc_min = 1e-12                  # Minimal value of sc, if it drops below this threshold the gradient descent stops.
                numIterations = 150000          # Maximal number of gradient descent steps - since the convergence is linear the required number can be large (thousands).

                
                # Run model and fit params for n iterations --------------------------------------------------------------------------

                E_list = [] # store all E-values for a range of parameters from original values to original values + 50*(0.5)
                A1_list = [] # hold A1 params in order
                A2_list = [] # hold A2 params in order
                A3_list = [] # hold A3 params in order 
                A4_list = [] # hold A4 params in order
                amp_list = [] # hold all amp params in order 
                lam_list = [] # hold all lam params in order
                e0_list = [] # hold all e0 params in order
                avg_list = []
                shift_list = []

                print('Starting gradient descent...')
                E_cur = Ecalc(A1,A2,A3,A4,lam,amp,e0,avg,shift,e_original,tstep,h,a,numSteps,sig)
                
                for i in range(numIterations):
                        #print(('%f  '*10) % (A1,A2,A3,A4,lam,amp,e0,avg,shift,E_cur))

                        # Generate the FD approximation for gradients for each variables:
                        grads = gradients(delta,e_original,A1,A2,A3,A4,lam,tstep,h,a,numSteps,amp,e0,avg,shift,sig, set_zero_idx=[2,3,7])
                        dE = list(itemgetter("dEdA1", "dEdA2", "dEdA3", "dEdA4", "dEdlam","dEdamp","dEde0", "dEdavg", "dEdshift")(grads))
                        dE = np.asarray(dE)
                        dE = sc * dE/np.linalg.norm(dE,2)

                        # The "t" stands for tentative. See how the cost function decreases with this step:
                        A1t = A1 + sc * dE[0]
                        A2t = A2 + sc * dE[1]
                        A3t = max(A3 + sc * dE[2],0.0)
                        A4t = A4 + sc * dE[3]
                        lamt = min(max(lam + sc * dE[4],0.1),10.0)
                        ampt = amp + sc * dE[5]
                        e0t = e0 + sc * dE[6]
                        avgt = avg + sc * dE[7]
                        shiftt = shift + sc * dE[8]
                        E = Ecalc(A1t,A2t,A3t,A4t,lamt,ampt,e0t,avgt,shiftt,e_original,tstep,h,a,numSteps,sig)

                        # Decision time: if the cost function decreases, accept the step and move on.
                        if E < E_cur: # Note: experiment with a stricter criterion, e.g. E < beta * E_cur for some 0 <= beta <= 1.
                                A1 = copy(A1t)
                                A1_list.append(A1)
                                A2 = copy(A2t)
                                A2_list.append(A2)
                                A3 = copy(A3t)
                                A3_list.append(A3)
                                A4 = copy(A4t)
                                A4_list.append(A4)
                                lam = copy(lamt)
                                lam_list.append(lam)
                                amp = copy(ampt)
                                amp_list.append(amp)
                                e0 = copy(e0t)
                                e0_list.append(e0)
                                avg = copy(avgt)
                                avg_list.append(avg)
                                shift = copy(shiftt)
                                shift_list.append(shiftt)
                                sc *= 1.5                # If the step was successful, increase the step size!
                                E_cur = copy(E)
                                E_list.append(E)
                        else:   
                                sc *= 0.5                # If the cost function did not decrease, decrease the step size.

                        # If the step size is too small, exit.
                        if sc < sc_min:
                                print('Trust region too small, exiting...')
                                break

                it_num = np.shape(E_list)[0]
                print("Number of iterations = %d" % (it_num))
                A1 = A1_list[it_num-1]
                A2 = A2_list[it_num-1]
                A3 = A3_list[it_num-1]
                A4 = A4_list[it_num-1]
                lam = lam_list[it_num-1]
                amp = amp_list[it_num-1]
                e0 = e0_list[it_num-1]
                avg = avg_list[it_num-1]
                shift = shift_list[it_num-1]

                # Re-create the result for plotting.
                intake = createInputPeak(numSteps,shift,sig,amp)
                u = u_calc(e_original,A1,A2,lam,tstep)
                e = e_calc(u,A3,A4,h,a,numSteps,amp,e0,avg,shift,sig)

                return {'e_original':e_original, 
                        'u':u,
                        'e':e,
                        'A1':A1,
                        'A2':A2,
                        'A3':A3,
                        'A4':A4,
                        'lam':lam, 
                        'amp':amp,
                        'e0':e0,
                        'shift':shift,
                        'E': E,
                        'zeros':zeros,
                        'A1_list':A1_list,
                        'A2_list':A2_list,
                        'A3_list':A3_list,
                        'shift_list':shift_list,
                        'lam_list':lam_list,
                        'amp_list':amp_list,
                        'e0_list':e0_list,
                        'avg_list':avg_list,
                        'E_list':E_list,
                        'intake':intake }

        
        
        def calculate_params_for_peaks(self, peak_keys = []):
                """
                Default is to calculate params for all peaks stored in self.peaks. 
                If only a set of peaks to calculate, specify the peak keys in 'peak_keys' argument. 
                The the peak key: 'g_original', 'peak0', 'peak1', ... which is stored in peak.yaml.
                """
        
                if len(peak_keys) == 0:
                        peak_keys = self.peaks.keys()

                for p in peak_keys:
                        normalized_glucose = self.peaks[p]['values']
                        stdv = self.stdv / self.peaks[p]['max_val']

                        # Compute the params
                        grad_params = self.gradientDescent(normalized_glucose)
                        self.params_log[p] = grad_params

                        # Retrieve the params
                        e_original, u, e, A1, A2, A3, A4, lam, amp, e0, shift, E, zeros, A1_list, A2_list, A3_list, shift_list, lam_list, amp_list, e0_list, avg_list, E_list, intake = \
                        itemgetter('e_original', 'u', 'e', 'A1', 'A2', 'A3', 'A4', 'lam', 'amp', 'e0', 'shift', 'E', 'zeros', 'A1_list', 'A2_list', 'A3_list', 'shift_list', 'lam_list', 'amp_list', 'e0_list', 'avg_list', 'E_list', 'intake')(grad_params)

                        # Save params to dict
                        self.params[p] = {}
                        self.params[p]['u'] = u.tolist()
                        self.params[p]['E'] = E
                        self.params[p]['A1'] = A1.item()
                        self.params[p]['A2'] = A2.item()
                        self.params[p]['A3'] = A3.item()
                        self.params[p]['A4'] = A4.item()
                        self.params[p]['lam'] = lam.item()
                        self.params[p]['amp'] = amp.item()
                        self.params[p]['shift'] = shift.item()
                        self.params[p]['e0'] = e0.item()
                        self.params[p]['A1/A2'] = (A1/A2).item()
                        self.params[p]['stdv*A1/lam'] = (stdv*A1/lam).item()
                        self.params[p]['stdv*A2/lam'] = (stdv*A2/lam).item()
                        self.params[p]['stdv*(A1+A2)/lam'] = (stdv*(A1+A2)/lam).item()
                        self.params[p]['A1 * A3/lam**2'] = (A1 * A3/lam**2).item()
                        self.params[p]['A2 * A3/lam**2'] = (A2 * A3/lam**2).item()
                        self.params[p]['(A1+A2) * A3/lam**2'] = ((A1+A2) * A3/lam**2).item()
                        self.params[p]['np.max(u)/lam'] = (np.max(u)/lam).item()
                        self.params[p]['np.max(u)/(stdv*(A1+A2))'] = (np.max(u)/(stdv*(A1+A2))).item()


                return self.params


        def plot_model_fitting_with_params(self, param_keys = []):
                        
                """
                Default is to print out all params for different peaks (i.e., manual selected/ machine selected) in one figure.
                If one want to specify certain params to plot, adds the param key to the argument "param_keys". 
                The param key is identical to the peak key, i.e., 'g_original', 'peak0', 'peak1', ... and so on.
                """

                if len(param_keys) == 0:
                        param_keys = self.params.keys()

                
                plt.figure(figsize=[10*2, 10*(len(param_keys)/2)])
                
                auto_peaks_key = [k for k in param_keys if k.startswith('peak_')]
                subplot_grid_base = math.ceil((len(auto_peaks_key) + 2)/2) * 100 + 20


                for n, p in enumerate(param_keys):
                        
                        if p == 'g_original':
                                idx = 1
                        elif ('g_original' in param_keys):
                                idx = 2 + n
                        else:
                                idx = 3 + n
                        plt.subplot(subplot_grid_base + idx)
                        

                        e_original = self.params_log[p]['e_original']
                        e = self.params_log[p]['e']
                        u = self.params_log[p]['u']
                        intake = self.params_log[p]['intake']
                        
                        A1 = self.params[p]['A1']
                        A2 = self.params[p]['A2']
                        lam = self.params[p]['lam']

                        # The plots are saved in postscript format.
                        Nplot = 60 # Magic number alert! This is the number of points used for plotting.
                        x_list = range(len(e_original))
                        x_arr = np.array(x_list)
                        x_smooth = np.linspace(x_arr.min(), x_arr.max(),Nplot)

                        y1 = CubicSpline(x_list, e_original)
                        y2 = CubicSpline(x_list, e)
                        y3 = CubicSpline(x_list, u)
                        y4 = CubicSpline(x_list, intake)

                        plt.plot(x_smooth,y1(x_smooth),label='subject data')
                        plt.plot(x_smooth,y2(x_smooth),label='model data')
                        plt.plot(x_smooth,y3(x_smooth)/(A1+A2),label='controller')
                        plt.plot(x_smooth,y4(x_smooth)/lam,label='intake')

                        if idx == 1:
                                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                        
                        plt.xticks(fontsize=6)
                        plt.yticks(fontsize=6)
                        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.3, wspace=0.2)
                        plt.title("params for:" + p, fontsize=8)
                        plt.grid(True)

                output_dir = Path(self.log_dir)
                output_path = output_dir / 'out_{}.png'.format(self.subject_id)
                plt.savefig(output_path,format='png')
                plt.close('all')
                

        def plot_error_over_time(self, param_keys = []):
                
                """
                Default is to print out all params for different peaks (i.e., manual selected/ machine selected) in one figure.
                If one want to specify certain params to plot, adds the param key to the argument "param_keys". 
                The param key is identical to the peak key, i.e., 'g_original', 'peak0', 'peak1', ... and so on.
                """

                if len(param_keys) == 0:
                        param_keys = self.params.keys()

                plt.figure(figsize=[10*2, 10*(len(param_keys)/2)])
                
                auto_peaks_key = [k for k in param_keys if k.startswith('peak_')]
                subplot_grid_base = math.ceil((len(auto_peaks_key) + 2)/2) * 100 + 20



                for n, p in enumerate(param_keys):

                        if p == 'g_original':
                                idx = 1
                        elif ('g_original' in param_keys):
                                idx = 2 + n
                        else:
                                idx = 3 + n
                        plt.subplot(subplot_grid_base + idx)


                        E_list = self.params_log[p]['E_list']
                        e_original = self.params_log[p]['e_original']
                        zeros = self.params_log[p]['zeros']

                        for i in range(len(E_list)):
                                E_list[i] = E_list[i]/sse(e_original,zeros)

                        plt.semilogy(np.arange(len(E_list)),E_list)
                        plt.xlabel('Number of iteration', fontsize=6)
                        plt.ylabel('Error', fontsize=6)

                        if idx == 1:
                                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                        
                        plt.xticks(fontsize=6)
                        plt.yticks(fontsize=6)
                        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.3, wspace=0.2)
                        plt.title("params for:" + p, fontsize=8)
                        plt.grid(True)

                output_dir = Path(self.log_dir)
                err_path = output_dir / 'err_{}.png'.format(self.subject_id)
                plt.savefig(err_path,format='png')
                plt.close('all')


        def plot_params_over_time(self, param_keys = []):
                """
                Default is to print out all params for different peaks (i.e., manual selected/ machine selected) in one figure.
                If one want to specify certain params to plot, adds the param key to the argument "param_keys". 
                The param key is identical to the peak key, i.e., 'g_original', 'peak0', 'peak1', ... and so on.
                """

                if len(param_keys) == 0:
                        param_keys = self.params.keys()

                plt.figure(figsize=[10*2, 10*(len(param_keys)/2)])
                
                auto_peaks_key = [k for k in param_keys if k.startswith('peak_')]
                subplot_grid_base = math.ceil((len(auto_peaks_key) + 2)/2) * 100 + 20

                for n, p in enumerate(param_keys):

                        if p == 'g_original':
                                idx = 1
                        elif ('g_original' in param_keys):
                                idx = 2 + n
                        else:
                                idx = 3 + n
                        plt.subplot(subplot_grid_base + idx)

                        A1_list = self.params_log[p]['A1_list']
                        A2_list = self.params_log[p]['A2_list']
                        A3_list = self.params_log[p]['A3_list']
                        lam_list = self.params_log[p]['lam_list']
                        amp_list = self.params_log[p]['amp_list']
                        e0_list = self.params_log[p]['e0_list']
                        avg_list = self.params_log[p]['avg_list']
                        shift_list = self.params_log[p]['shift_list']

                        x_listP = np.arange(len(A1_list))

                        plt.plot(x_listP,A1_list,label='A1')
                        plt.plot(x_listP,A2_list,label='A2')
                        plt.plot(x_listP,A3_list,label='A3')
                        plt.plot(x_listP,lam_list,label='lambda')
                        plt.plot(x_listP,amp_list,label='amp')
                        plt.plot(x_listP,e0_list,label='e0')
                        plt.plot(x_listP,avg_list,label='ebar')
                        plt.plot(x_listP,shift_list,label='shift')


                        if idx == 1:
                                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                        
                        plt.xticks(fontsize=6)
                        plt.yticks(fontsize=6)
                        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.3, wspace=0.2)
                        plt.title("params for:" + p, fontsize=8)
                        plt.grid(True)

                output_dir = Path(self.log_dir)
                parsplot_path = output_dir / 'pars_{}.png'.format(self.subject_id)
                plt.savefig(parsplot_path,format='png')

                plt.close('all')


        def A1_A2_undimentionalized_for_peaks(self, param_keys = []):
                if len(param_keys) == 0:
                        param_keys = self.params.keys()

                manual_color = 'mediumvioletred'
                auto_colors = ['steelblue', 'cadetblue', 'deepskyblue', 'aquamarine', 'cornflowerblue']
                color_code = {}

                plt.figure(figsize=[10, 10*(len(param_keys))])
                
                subplot_grid_base = (len(param_keys)+1) * 100 + 10

                auto_peak_idx = 0
                auto_A1_sum = 0
                auto_A2_sum = 0

                for p in param_keys:
                        
                        idx = 1
                        plt.subplot(subplot_grid_base + idx)

                        if p == 'g_original':
                                color_code[p] = manual_color
                        else:
                                color_code[p] = auto_colors[auto_peak_idx % 5]
                                auto_peak_idx += 1
                        
                        A1 = self.params[p]['A1']
                        A2 = self.params[p]['A2']
                        max_glucose = self.peaks[p]['max_val']
                        max_u = max(self.params[p]['u'])
                        stdv = self.stdv / max_glucose

                        x = stdv * (A1 / max_u)
                        y = stdv * (A2 / max_u)

                        self.params[p]['undimentional_A1'] = x
                        self.params[p]['undimentional_A2'] = y

                        if p != 'g_original':
                                auto_A1_sum += x
                                auto_A2_sum += y

                        plt.text(x-0.01, y+0.02, p)
                        plt.plot(x, y, color=color_code[p], marker='o')
                        
                
                # Plot the average auto params
                avg_A1 = auto_A1_sum/auto_peak_idx
                avg_A2 = auto_A2_sum/auto_peak_idx
                plt.plot(avg_A1, avg_A2, color='y', marker='o')
                plt.text(avg_A1-0.01, avg_A2+0.02, 'avg_auto')

                self.params['auto_avg']['undimentional_A1'] = avg_A1
                self.params['auto_avg']['undimentional_A2'] = avg_A2

                for i, p in enumerate(param_keys):
                        idx = 2+i
                        plt.subplot(subplot_grid_base + idx)

                        peak_glucose = self.peaks[p]['values']
                        
                        plt.plot(np.arange(len(peak_glucose)), peak_glucose, '|-', color=color_code[p])

                        plt.title("peak:" + p, fontsize=8)



                output_dir = Path(self.log_dir)
                parsplot_path = output_dir / 'A1_A2_{}.png'.format(self.subject_id)
                plt.savefig(parsplot_path,format='png')

                plt.close('all')


        def plot_radarChart_compare_A1_A2_lamda_among_manual_auto_peak(self):
                """
                This function is to plot the major params comparison among manual selected peak and auto selected peaks.
                """

                data = []
                legend_label = []

                # Manual peak: The target to compare with
                if 'g_original' not in self.params:
                        data.append([0, 0, 0])
                else:
                        p = 'g_original'
                        data.append([self.params[p]['A1'], self.params[p]['A2'], self.params[p]['lam']])
                        legend_label.append(p)

                # Auto peaks
                auto_peak = [key for key in self.params if key.startswith('peak_')]

                for p in auto_peak:
                        data.append([self.params[p]['A1'], self.params[p]['A2'], self.params[p]['lam']])
                        legend_label.append(p)
                
                # Construct the data for Radar Chart
                data_to_plot = (['A1', 'A2', 'lambda'], data)


                fig, ax = radarplot.createFigure('Manual_vs_Auto_Peaks_params', 3, data_to_plot, 
                                                legend_label, 
                                                rgrid=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
                                                colors=['mediumvioletred', 'steelblue', 'cadetblue', 'deepskyblue', 'aquamarine', 'cornflowerblue'])
                
                #plt.subplots_adjust(bottom=0.5)

                output_dir = Path(self.log_dir)
                radarplot_path = output_dir / 'indiv_params_{}.png'.format(self.subject_id)
                fig.savefig(radarplot_path,format='png')

                plt.close('all')

def float_representer(dumper, value):
        text = '{0:.9f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)

def run_batch_and_save_params_to_yaml(peak_yaml_filename, peak_yaml_dir, log_output_dir):
    
    peak_dir = Path(peak_yaml_dir)
    peak_file = peak_dir / peak_yaml_filename
    subjects_peaks = read_peak(peak_file)

    print("number of subject:", len(subjects_peaks.keys()))

    # The main dict that will be saved to params yaml file
    batch_param_data = []
    count = 0

    for subject_id in subjects_peaks:
        print("===========================================================")
        print("Start processing:", subject_id)
        print("===========================================================")
        
        count += 1
        subj = subjectFitToModel(subject_id, subjects_peaks[subject_id], log_output_dir)

        # Each subject has a personal dict that will be added to batch_peak_data
        subject_data = {}
        subject_data['subject'] = subj.subject_id

        # Add stats
        subject_data['avg'] = subj.avg
        subject_data['stdv'] = subj.stdv

        # Run Gradient Descent for each peaks
        subject_data['params'] = subj.calculate_params_for_peaks()
        
        # Plot params to log
        # subj.plot_model_fitting_with_params()
        # subj.plot_error_over_time()
        # subj.plot_params_over_time()
        # subj.A1_A2_undimentionalized_for_peaks()
        # subj.plot_radarChart_compare_A1_A2_lamda_among_manual_auto_peak()

        # Add subjecct data to batch
        batch_param_data.append(subject_data)

        print("===subject data dict===================")
        print(subject_data)


    output_param_dict = {'SUBJECTS': {'DATA' : batch_param_data}}
    output_yaml = log_output_dir/ "params.yaml"
    
    # To restrict the float precision in order to write to YAML
    yaml.add_representer(float, float_representer)

    with open(output_yaml, 'w') as yaml_file:
        yaml.dump(output_param_dict, yaml_file, default_flow_style=False, sort_keys=False)

    print("total subj:", count)

def batch_FitToModel():
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
    

    # Setup sub dir for each experiment config 
    print(args.cfg)
    cfg_name = os.path.basename(args.cfg).split('.')[0]
#     time_str = time.strftime('%Y_%m_%d_%H_%M')
#     final_output_dir = output_dir / cfg_name / (cfg_name + '_peak_' + time_str)
    final_output_dir = output_dir / cfg_name / (cfg_name + '_peak')
    print(final_output_dir)
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)


    ######################################################       
    # Run params extraction
    ######################################################
    run_batch_and_save_params_to_yaml(peak_yaml_filename, peak_dir, final_output_dir)



if __name__ == "__main__":
        ######################################################
        # Config
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
        
        # Setup Log dir
        output_dir = Path(cfg.LOG_DIR)
        if not output_dir.exists():
                print('=> creating {}'.format(output_dir))
                output_dir.mkdir()

        # Setup sub dir for each experiment config 
        cfg_name = os.path.basename(args.cfg).split('.')[0]
        time_str = time.strftime('%Y_%m_%d_%H_%M')
        final_output_dir = output_dir / cfg_name / (cfg_name + '_grad_' + time_str)
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)


        ######################################################       
         # Read subjects' peaks data
        ######################################################
        peak_dir = Path(cfg.PEAK_DATA_DIR)
        peak_file = cfg.PEAK_DATA_FILE
        peak_file = peak_dir / peak_file
        subject_peaks = read_peak(peak_file)
  
        # ######################################################
        #  # Read subject range from config
        # ######################################################
        
        # create an instance of subjectFitToModel
        subject_id = '0M00048323W'
        subject_1 = subjectFitToModel(subject_id, subject_peaks[subject_id], final_output_dir)

        # show subject stats
        print("subject id:", subject_1.subject_id)
        print("subject glucose average:", subject_1.avg)
        print("subject standard deviation:", subject_1.stdv)
        
        # show subject's peak data
        print("subject's peak data")
        pprint.pprint(subject_1.peaks)

        # start calculate the params for each peak
        print("Starting gradient Descent for each peak...")
        subject_1.calculate_params_for_peaks()

        # show the params:
        print("subject's params:")
        pprint.pprint(subject_1.params)

        # save plot of the params and params_log:
        subject_1.plot_model_fitting_with_params()
        subject_1.plot_error_over_time()
        subject_1.plot_params_over_time()
        subject_1.plot_radarChart_compare_A1_A2_lamda_among_manual_auto_peak()
