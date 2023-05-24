# Returns the input peak values. Shape is assumed to be Gaussian.
import numpy as np

# Gaussian peak with amplitude ampl, standard deviation sig centered at t = shift
def peak(t,shift,sig,amp):
        F = amp * np.exp(-(t-shift)**2/(2.0 * sig**2)) / np.sqrt(2.0 * np.pi * sig**2)
        
        return F
