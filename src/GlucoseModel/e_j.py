# Just a function that returns an element e[j] for j >=0 and 0 for j<0.
# This means we assume that e=0 before the peak, i.e. for the purpose of computing the integral term for u.
def e_j(e,j):
        if j < 0:
                return 0.0
        else:
                return e[j]
