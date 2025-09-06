# Required import
import numpy as np



# Gaussian function
# Parameters:
# - x     : independent variable (scalar or array)
# - x0    : mean (center of the distribution)
# - sigma : standard deviation (spread of the distribution)
# Returns the normalized Gaussian value evaluated at x
def gaussian(x, x0, sigma):
    return 1/(np.sqrt(2*np.pi) * sigma) * np.exp(-0.5 * ((x - x0) / sigma)**2)



# Piecewise constant function
# Parameters:
# - x   : independent variable (scalar or array)
# - c1  : constant value for x < L/2
# - c2  : constant value for x >= L/2
# - L   : reference length for the threshold
# Returns an array of values equal to c1 or c2 depending on the condition
def c_piecewise(x, c1, c2, L):
    return np.where(x < L/2, c1, c2)