# Computes the sum of squares of differences between elements of a and b.
# I suppose this can be replaced by a call to np.linalg.norm(a-b,2)**2 if a and b have the right type?
import math

def sse(a,b):
	sse_tot = 0
	#if vectors not same length, cannot determine
	if(len(a) != len(b)):
		return "Vectors not same length"
	else:
		for i in range(len(a)):
			try:
				sse_tot = sse_tot + math.pow(abs(a[i]-b[i]),2)
			except:
				sse_tot = float('inf') # More clunky exception handling...
	return sse_tot
