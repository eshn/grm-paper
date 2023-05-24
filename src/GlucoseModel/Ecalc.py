# Compute the cost function.
from u_calc import u_calc
from e_calc import e_calc
from sse import sse
import numpy as np

def Ecalc(A1,A2,A3,A4,lam,amp,e0,avg,shift,e,tstep,h,a,numSteps,sig):
	#1) Get original u
	u_original = u_calc(e,A1,A2,lam,tstep)
	#2) Get calculated e
	e_calculated = e_calc(u_original,A3,A4,h,a,numSteps,amp,e0,avg,shift,sig)
	#3) Get E
	E = sse(e, e_calculated)

	return E
