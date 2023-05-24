# Returns the time series F(t_k), i.e. the input peak.
from peak import peak

def createInputPeak(numSteps,shift,sig,amp):
	input = []
	for i in range(numSteps):

		# peak() is to get Gaussian 
		input.append(peak(float(i),shift,sig,amp))

	return input
