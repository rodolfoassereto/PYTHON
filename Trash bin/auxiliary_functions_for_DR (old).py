import numpy as np
from differential_operators import *

### FUNCTIONS FOR DOUGLAS-RACHFORD ###

def II(u):
    return np.sum(u, axis=(-1,-2))

def IIstar(ubar, q1, q2): # uses array broadcasting rule as a trick
    aux = np.zeros(ubar.shape + (q1,q2))
    return np.transpose(np.transpose(aux) + np.transpose(ubar))

def JJ(u):
    size = np.size(u[0,0])
    return np.transpose(np.transpose(u) - np.transpose(II(u))/size)

def JJ2(v):
    return np.array([JJ(v[0]), JJ(v[1])])

def nabla2_q(v):
    return np.array([ nabla_q(v[0]), nabla_q(v[1]) ]) # remark: il primo indice scorre "I_2" (prod. tensore)

def div2_q(gamma):
    return np.array([div_q(gamma[0]), div_q(gamma[1])])

######################## From the balanced version:

def UU(u, mask):
    return u[mask]
    # return u[..., mask] # old version: works with 2d undersampling mask # mask is (Q1,Q2) and u is (R1,R2,Q1,Q2)

def UUstar(lam, mask):
    out = np.zeros(mask.shape, dtype=complex)
    out[mask] = lam
    # The following works for a 2d undersampling mask
    # out = np.zeros(lam.shape[0:2] + mask.shape, dtype=complex)
    # out[..., mask] = lam
    return out

def KK(u, mask):
    # return UU(u, mask)
    return UU(np.fft.fftn(u, norm='ortho'), mask)

def KKstar(lam, mask):
    # return np.real(UUstar(lam, mask))
    return np.real(np.fft.ifftn(UUstar(lam, mask), norm='ortho'))

########################

def LL(u, gamma, mask, alp, bet):
    v = alp * JJ2(nabla_r(u)) - div2_q(gamma) # note: should be +gamma, but actually gamma is a free variable
    lam = KK(u, mask)
    eta = alp*bet * nabla_r(II(u)) # remark: nabla_r acts as nabla on a 2d array
    return (v, lam, eta)

def LLstar(v, lam, eta, mask, alp, bet):
    u = -alp*div_r(JJ2(v)) + KKstar(lam, mask) - alp*bet * IIstar(div_r(eta), v.shape[-2], v.shape[-1])
    gamma = nabla2_q(v)
    return (u, gamma)

def M_inv(u, gamma, xi, precond='Richardson'):
    if precond == 'Richardson': return (u/xi, gamma/xi)
    return # Other precondiioners...
def T(u, gamma, sig, tau, mask, alp, bet):
    tempu, tempgamma = LLstar( *LL( u, gamma, mask, alp, bet ), mask, alp, bet )
    return (u + sig*tau*tempu, gamma + sig*tau*tempgamma)

### Here I define the linear operator for the standard-TV problem

def LL1(u,gamma,mask,alp):
    v = alp * nabla_r(u) + gamma
    lam = KK(u, mask)
    return (v, lam)
def LL1star(v, lam, mask, alp):
    u = -alp*div_r(v) + KKstar(lam, mask)
    gamma = v.copy()
    return (u, gamma)
def T1(u, gamma, sig, tau, mask, alp):
    tempu, tempgamma = LL1star( *LL1( u, gamma, mask, alp), mask, alp )
    return (u + sig*tau*tempu, gamma + sig*tau*tempgamma)