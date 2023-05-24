# For input time series e(t_k) compute first u(t_k) and then the "output" e(t_k).
from u_calc import u_calc
from e_calc import e_calc

def e_get(e,A1,A2,A3,A4,lam,tstep,h,a,numSteps,amp,e0,avg,shift,sig):
	#1) Get original u
	u_original = u_calc(e,A1,A2,lam,tstep)

	#2) Get calculated e
	e_calculated = e_calc(u_original,A3,A4,h,a,numSteps,amp,e0,avg,shift,sig)

	return e_calculated
