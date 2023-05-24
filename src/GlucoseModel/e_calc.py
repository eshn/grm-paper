# For a given time series u(t_k) compute the time series e(t_k). Uses explicit Euler.
from peak import peak

def e_calc(u,A3,A4,h,a,numSteps,amp,e0,avg,shift,sig):
	e_calc = []

	e_calc.append(e0)
	for t in range(numSteps-1):
		try:
			e_calc.append(e_calc[t] + h * ( - A3 - A4 * u[t] * (e_calc[t]+avg) + peak(float(t),shift,sig,amp) ) )
		except:
			e_calc.append(float('inf')) # Bad form of exception handling! This must be done more robustly.
			                            
	return e_calc
