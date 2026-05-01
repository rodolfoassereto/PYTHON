''' This file contains the discrete differential operators'''

import numpy as np

def partialyplus(x): # ! Bredies convention is the opposite,(ie: partialy are the horizontal differences)
    n = x.shape[1]
    v = np.zeros((1, n))
    y = np.r_[x, v] # appende v in fondo a x (in basso!)
    z = y[1:] - x
    z[-1] = v
    return z

def partialxplus(x):
    return np.transpose( partialyplus( np.transpose(x) ) )

def nabla(x):
    return np.r_[ partialxplus(x)[None,...], partialyplus(x)[None,...] ]

def TV(x):
    return np.sum(np.linalg.norm((partialxplus(x), partialyplus(x)), axis=0))

###############################################################################

def partialyminus(x): # ! Bredies convention is the opposite
    z = x.copy()
    z[1:] = z[1:] - z[:-1]
    z[-1] = -x[-2]
    return z

def partialxminus(x):
    return np.transpose(partialyminus(np.transpose(x)))

def nablastar(arr): # arr is meant to be ( partialxplus(x), partialyplus(x) )
    return - partialxminus(arr[0]) - partialyminus(arr[1])

###############################################################################

def nablastarnabla(x):
    return - partialxminus(partialxplus(x)) - partialyminus(partialyplus(x))