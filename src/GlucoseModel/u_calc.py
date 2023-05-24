
# Compute the control variable for a given time series e(t) (excess glucose concentration).
import math
import numpy as np

# Weight function. Note: defined with a factor of lambda so that it is normalized to unity.
def w(tau,lam):
        return lam * np.exp(-lam * tau)

# Input e. To avoid the default circular indexing of Python, we return 0 for data before or after the recorded time series.
def input_e(e,k):
        if k < 0:
                return 0.0
        elif k < len(e):
                return e[k]
        else:
                return 0.0
        
def u_calc(e,A1,A2,lam,tstep):
        u_orig = [] #values for u_original placed here 
	
        # Apply left point rule to approximate integral.
        # Number of panels (integral from t-2/lam to t):
        M = int(math.ceil(2.0/(lam*tstep))) # Comment: re-computed for every new call so that the domain of integration is always t - 2/(lambda) .. t
        
        for i in range(len(e)):
                lp_sum = 0.0   # Sum of all mp_rule terms for integral approximation.
                for j in range(M):
                        lp_sum = lp_sum + A2 * tstep * w(float(M-j)*tstep,lam) * input_e(e,i+j-M)
                u_orig.append(lp_sum)
        u_orig = np.asarray(u_orig)
        
	# Add A1*e term to the u-value to get u_original.
        for i in range(len(e)):
                u_orig[i] = u_orig[i] + A1*e[i]

        return u_orig
