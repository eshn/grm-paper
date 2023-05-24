# Given a representative peak g, subtract the set point SP and add zeros at the end.
# The result, e, is used as target for the optimization (gradient descent).
def convertGtoE(g,SP,zeroPadding):
	e = []
	
	for i in range(len(g)):
		e.append((g[i] - SP))
	
	for i in range(zeroPadding):
		e.append(0.0)

	return e
