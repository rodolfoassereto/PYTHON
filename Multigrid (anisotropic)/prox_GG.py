'''
This file computes the second prox, i.e. projects on the epigraph of g and h
- "g", "prox_g"
- "find_t" finds the solution to the fixed point problem given from the projection on epi(g)
- "proj_epi_g"
- "h", "prox_1norm", "prox_h"
- "find_tau" finds the solution to the fixed point problem given from the projection on epi(h)
- "F" and "plot_F" are the (respectively) the piecewise affine function that arises from the fixed point problem 
from "proj_epi_h", and its plot
- "prox_GG" is the seeked prox
'''

import numpy as np
import matplotlib.pyplot as plt

##### epi(g)

def g(i, u, z): # i is the "depth" of u
    return 0.5 * ( 2**i * np.linalg.norm(u - z) )**2

def prox_g(fac, i, ubar, z):
    return (ubar + fac*2**(2*i)*z)/(1+fac*2**(2*i))

def find_t(i, vbar, z, tbar):
    p = np.array([2**(4*i),
         2**(2*i+1)-2**(4*i+1)*tbar,
         2**(4*i)*tbar**2 - 2**(2*i+1)*tbar + 1,
         -2**(2*i-1)*np.linalg.norm(vbar-z)**2])
    ts = np.roots(p)
    real_ts = [x.real for x in ts if x.imag == 0]
    return real_ts[0]
    
def proj_epi_g(i, vbar, z, tbar): # i is the "depth" of v
    if g(i, vbar, z) <= tbar: return (vbar, tbar) # !! Note: this never happens in my experiment
    t_tilde = find_t(i, vbar, z, tbar)
    print(np.abs( g(i, prox_g(t_tilde-tbar, i, vbar, z), z) - t_tilde ))
    return (prox_g(t_tilde-tbar, i, vbar, z), t_tilde)


##### epi(h) (for anisotropic TV)

def h(alpha, i, w):
    return (2**i) * alpha * np.sum(np.abs(w))

def prox_1norm(gamma, mat):
    return (np.abs(mat) - gamma).clip(min=0) * np.sign(mat)

def prox_h(gamma, i, alpha, w):
    if np.abs(gamma) < 1e-12: return w
    return prox_1norm(2**i * alpha * gamma, w)

def find_tau(i, alpha, d): # this works for both iso and anisotropic
    # d = np.matrix.flatten(C) # Note: if you don't flatten, you shouldn't use "ord=np.inf"
    # tau = np.linalg.norm(d, ord=np.inf)/2
    tau = np.max(np.max(d)/2,0)
    beta = 2**(2*i) * alpha**2
    f = lambda x, t: beta * np.sum((x-t).clip(min=0))
    n = f(d, tau)
    while np.abs(n-tau)>1e-6:
        m = beta * np.sum(d>tau)
        tau = (n+m*tau)/(m+1) # note: m is the POSITIVE slope
        n = f(d,tau)
    return tau

def proj_epi_h(i, alpha, wbar, taubar): # (!!) It often happens that taubar<0: can this generate problems?
    if h(alpha, i, wbar) <= taubar: return (wbar, taubar) # !! Note: this never happens in my experiment
    c = taubar + np.abs(wbar)/(alpha* 2**i)
    tau_tilde = find_tau(i, alpha, c)
    # if tau_tilde < taubar: print('Minore!')
    return (prox_h(tau_tilde-taubar, i, alpha, wbar), tau_tilde)

##### Extra functions

def F(x, t, i, alpha):
    beta = 2**(2*i) * alpha**2
    return beta * np.sum((x-t).clip(min=0))

def plot_F(wbar, i, alpha, taubar, minval=None, maxval=None):
    d = np.abs(wbar)/(2*i * alpha) + taubar
    if minval != 0 and not(minval): minval = np.min(d)
    if maxval != 0 and not(maxval): maxval = np.max(d)
    xx = np.linspace(minval, maxval, int(1e5))
    yy = np.array([F(d, tau, i, alpha) for tau in xx])
    plt.plot(xx,yy)
    plt.show()
    return

########## GG

def prox_GG(sigma, alpha, vv, ww, tt, ttau, zz):
    l = len(zz)
    
    projG = [ proj_epi_g(i, vv[i]/sigma, zz[i], tt[i]/sigma) for i in range(l)]
    projH = [ proj_epi_h(i, alpha, ww[i]/sigma, ttau[i]/sigma) for i in range(l)]
    vv2 = np.empty(l, dtype=object)
    ww2 = np.empty(l, dtype=object)
    vv2[:] = [ vv[i] - sigma*projG[i][0]  for i in range(l) ]
    tt2 = np.array([ tt[i] - sigma*projG[i][1] for i in range(l) ])
    ww2[:] = [ ww[i] - sigma*projH[i][0] for i in range(l) ]
    ttau2 = np.array([ ttau[i] - sigma*projH[i][1] for i in range(l) ])
    
    # for i in range(l):
    #     x, t = projG[i]
    #     y, tau = projH[i]
    #     print('delta G = ',g(i, x, zz[i]) - t)
    #     print('delta H = ',h(alpha, i, y) - tau)
    
    # I_G = []
    # I_H = []
    # for i in range(l):
    #     if g(i, vv[i]/sigma, zz[i]) > tt[i]/sigma + 1e-5: I_G = I_G + [i]
    #     if h(alpha, i, ww[i]/sigma) > ttau[i]/sigma +1e-5: I_H = I_H + [i]
    # for i in I_G:
    #     x, t = projG[i]
    #     if np.abs(g(i, x, zz[i]) - t) > 1e-4: print('Errore in projG')
    # for i in I_H:
    #     y, tau = projH[i]
    #     if np.abs(h(alpha, i, y) - tau) > 1e-4: print('Errore in projH')
    
    return vv2, ww2, tt2, ttau2


# EXAMPLE FOR WHICH find_tau DOESN'T CONVERGE !!

# i, alpha = 1, 3
# wbar = 100*np.random.rand(2,32,32)-50
# tau = h(alpha, i, wbar)
# print('h(wbar) = ', tau)
# taubar = 0.9 * tau
# d = np.abs(wbar)/(2**i * alpha) + taubar
# print('max(d) = ',np.max(d))
# print('min(d) = ',np.min(d))

# plot_f(wbar, i, alpha, taubar)

# tau_tilde = find_tau(i, alpha, d)
# print(tau_tilde)
 