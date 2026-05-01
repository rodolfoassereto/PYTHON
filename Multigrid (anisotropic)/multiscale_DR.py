'''
This file performs 1 step of the preconditioned D-R.
- "KK" is the linear operator from the objective function
- "KKstar", "KKstarKK"
- "T" is the linear operator as defined in the D-R algorithm
- "Minv_precond" is the Richardson precondirioner (just divides by lambda)
- "multiscale_pDR_step" performs one step of pDR
'''

import numpy as np

import prox_FF as pf
import prox_GG as pg
import gradients_and_TV_multigrid as tv

def KK(uu,r,s):
    l = len(uu)
    ww_= np.empty(l, dtype=object)
    ww_[:] = [tv.nabla(x) for x in uu]
    return uu, ww_, np.array([r]*l), np.array([s]*l)

def KKstar(vv,ww,tt,ttau):
    temp = np.empty(len(vv), dtype=object)
    temp[:] = [ x + tv.nablastar(y) for x,y in zip(vv,ww) ]
    return temp, np.sum(tt), np.sum(ttau)

def KKstarKK(uu, r, s): # Note: for cycles in this function span on ~20 values at most (20 levels = 2^20 side)
    l = len(uu)
    temp1 = np.empty(l, dtype=object)
    temp1[:] = [ x + tv.nablastarnabla(x) for x in uu ]
    return temp1, l*r, l*s

def T(sigma1, sigma2, uu, r, s):
    uutemp, rtemp, stemp = KKstarKK(uu, r, s)
    fac = sigma1*sigma2
    return uu + fac*uutemp, r + fac*rtemp, s + fac*stemp 

def Minv_precond(uu, r, s, lam): # Will just use Richardson preconditioner at the moment
    return uu/lam, r/lam, s/lam


def multiscale_pDR_step(alpha, uu, r, s, vv, ww, tt, ttau, duu, dr, ds, baruu, barr, bars, barvv, barww, bartt, barttau, sigma1, sigma2, zz, lam): # lam is the lambda of the Richardson preconditioner
    l = len(zz)
    if l != len(uu): raise Exception('len(uu) is different from len(zz)')
    uu1, r1, s1 = pf.prox_FF(sigma1, baruu, barr, bars)
    
    vv1, ww1, tt1, ttau1 = pg.prox_GG(sigma2, alpha, barvv, barww, bartt, barttau, zz)
    
    ### b
    uu_temp, r_temp, s_temp = KKstar(2*vv1-barvv, 2*ww1-barww, 2*tt1-bartt, 2*ttau1-barttau)
    
    buu1 = 2*uu1-baruu - sigma1*uu_temp
    br1 = 2*r1-barr - sigma1*r_temp
    bs1 = 2*s1-bars - sigma1*s_temp
    
    ### d
    uu_temp, r_temp, s_temp = T(sigma1, sigma2, duu, dr, ds)
    uu_temp, r_temp, s_temp = Minv_precond(buu1 - uu_temp, br1 - r_temp, bs1 - s_temp, lam)
    duu1, dr1, ds1 = duu + uu_temp, dr + r_temp, ds + s_temp
    
    ### bar x^k+1
    baruu1 = baruu - uu1 + duu1
    barr1 = barr - r1 + dr1
    bars1 = bars - s1 + ds1
    
    ### bar y^k+1
    vv_temp, ww_temp, tt_temp, ttau_temp = KK(duu1,dr1,ds1)

    barvv1 = vv1 + sigma2*vv_temp
    barww1 = ww1 + sigma2*ww_temp
    bartt1 = tt1 + sigma2*tt_temp
    barttau1 = ttau1 + sigma2*ttau_temp
    
    return uu1, r1, s1, vv1, ww1, tt1, ttau1, duu1, dr1, ds1, baruu1, barr1, bars1, barvv1, barww1, bartt1, barttau1