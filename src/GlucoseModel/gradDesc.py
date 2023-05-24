# Copmute the finite difference approximation of components of the gradient.
import copy
import deprecation
from Ecalc import Ecalc

def gradients(delta,e,A1,A2,A3,A4,lam,tstep,h,a,numSteps,amp,e0,avg,shift,sig, set_zero_idx=[]):
	"""
	This function returns the gradients for each of the following parameter: 
		0. dEdA1
		1. dEdA2
		2. dEdA3
		3. dEdA4
		4. dEdlam
		5. dEdamp
		6. dEde0
		7. dEdavg
		8. dEdshift
	In case gradient=0 is desired for some of these parameters, simply specifies the index of which
	in "set_zero_idx" list. For example, if dEdA2 and dEdA3 are not desired, set "set_zero_idx = [1,2]".

	The return gradients is a dictionary: {"dEdA1": value, "dEdA2": value, "dEdA3": value, ..., "dEdshift": value}
	"""
	var_names = ["dEdA1", "dEdA2", "dEdA3", "dEdA4", "dEdlam","dEdamp","dEde0", "dEdavg", "dEdshift"]
	gradients = { gr:0.0 for gr in var_names}

	#params = [delta, e,A1,A2,A3,A4,lam,tstep,h,a,numSteps,amp,e0,avg,shift,sig]
	const = [e, tstep, h, a, numSteps, sig]
	variables = [A1,A2,A3,A4,lam,amp,e0,avg,shift]
	full_params = variables + const
	E0 = Ecalc(*full_params)


	#fd = max(delta * A3,1e-4)
	for i in range(len(variables)):
		perturbate_vars = copy.copy(variables)
 
		if i not in set_zero_idx:
			target_var = perturbate_vars[i]

			# Finite Difference scaler
			fd = delta * target_var
			if i in [2,6]: 		 # A3 and e0 has different scalar, as they starts off 0
				fd = max(delta * target_var, 1e-4)
			if target_var == 0:  # Ad-hoc for cases: shift = 0 
				fd = 1e-4

			# Finite diff: E1 - E0
			perturbate_vars[i] += fd
			full_params = perturbate_vars + const
			E1 = Ecalc(*full_params)
			finite_diff = E1 - E0
			#print("finite", finite_diff)
			
			# Scalar
			gradients[var_names[i]] =  ( -1 / fd ) * finite_diff
			#print("result:", gradients[var_names[i]])
	
	return gradients



