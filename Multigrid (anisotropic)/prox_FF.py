'''
This file computes the projection on the kernel of calligraphic S (i.e. smaller grids are the average of
bigger grids)
- "S_star" (S is not there because I used the function halve from building_data.py)
- "SS", "SS_star" (double S stands for calligraphic S)
- "alpha_seq", "gamma_seq" are auxiliary sequences of numbers used for the inversion of SS \circ SS_star
- "ww_i" is an auxiliary function 
- "invert_system_SS" invert the system SS(SS_star(xx))=bb
- Finally, "proj_ker" is the wanted projection on the kernel
'''

import sys
sys.path.append('C:\\Users\\rodol\\Google Drive\\PYHTON\\Multigrid Denoising Spyder')
import numpy as np
from building_data import halve

def S_star(u):
    (a, _) = np.shape(u)
    bigger_u = np.zeros((2*a, 2*a))
    bigger_u[0::2,0::2] = u
    bigger_u[1::2,0::2] = u
    bigger_u[0::2,1::2] = u
    bigger_u[1::2,1::2] = u
    return 0.25 * bigger_u

def SS(uu): # SS stands for calligraphic S
    l = len(uu)
    return np.array([ halve(uu[i]) - uu[i+1] for i in range(l-1) ], dtype=object)

def SS_star(uu_1): # SS stands for calligraphic S
    l = len(uu_1)
    return np.array([S_star(uu_1[0])] + [ S_star(uu_1[i+1]) - uu_1[i] for i in range(l-1) ] + [-uu_1[-1]], dtype=object)

# >>>>>>>>>> PROJECTION ON THE KERNEL <<<<<<<<<<

# both sequences are 82 long (82 is arbitrary, but it should be enough)
alpha_seq = [1, 1.25] # note: alpha_seq[0] is useless (and arbitrarily set to 1)
# for i in range(80):
#     alpha_seq += [alpha_seq[-1] - 1/(4*alpha_seq[-1])]
for i in range(80):
    alpha_seq += [alpha_seq[1] - 1/(4*alpha_seq[-1])]
gamma_seq = [1]
for i in range(1, 82):
    gamma_seq += [gamma_seq[-1]/alpha_seq[i]]

def ww_i(i,bb): # i shifts over bb (goes from 0 to k-1)
    if i == 0: return gamma_seq[1]/gamma_seq[0] * bb[0]
    s = 0
    for t in range(i+1):
        s = s + 1/gamma_seq[i-t] * halve(bb[i-t], times=t)
    return gamma_seq[i+1] * s
    
def invert_system_SS(bb):
    l = len(bb)
    vv = [ ww_i(l-1, bb) ]
    for i in reversed(range(l-1)):
        vv = [ 1/alpha_seq[i+1] * S_star(vv[0]) + ww_i(i,bb) ] + vv # vv[0] is because we are building vv in the cycle so it's the first entry
    return np.array(vv, dtype=object)

def proj_ker(uu):
    return uu - SS_star(invert_system_SS(SS(uu)))

# >>>>>>>>>> END <<<<<<<<<<

def prox_FF(sigma, uu, r, s): # x contains all the variables "uu, r, s"
    return proj_ker(uu), r - sigma, s - sigma
