import numpy as np
import matplotlib.pyplot as plt
from prox_and_proj import prox_norm21, prox_norm1
from build_toymodel import *
from auxiliary_functions_for_DR import *

# R1, R2, Q1, Q2 are imported from d2_build_toymodel_balanced

################################################

def DR(sig, tau, alp, bet, xi, ubar, gammabar, vbar, lambar, etabar, du, dgamma, b, mask,
       maxit, prints=2, saveall=False, name = 'noname_' , plotall=False, info=None, samples = 5):
    
    out = []
    dist = []
    disteucl = []
    
    R1, R2, Q1, Q2 = np.shape(ubar)
    
    for it in range(maxit):
        
        u = ubar.clip(min=0)
        gamma = prox_norm21(sig, gammabar, ax=1) # !! check the ax parameter
        
        v = vbar.copy()
        lam = (lambar - tau*b) / (1+tau) 
        eta = etabar.clip(min=-1, max=1)
        
        tempu, tempgamma = LLstar(2*v-vbar, 2*lam-lambar, 2*eta-etabar, mask, alp, bet)
        bu = 2 * u - ubar - sig * tempu
        bgamma = 2 * gamma - gammabar - sig * tempgamma
        
        tempbu, tempbgamma = T(du, dgamma, sig, tau, mask, alp, bet)
        tempdu, tempdgamma = M_inv(bu - tempbu, bgamma - tempbgamma, xi, precond='Richardson')
        du = du + tempdu 
        dgamma = dgamma + tempdgamma
        
        ubar = ubar - u + du
        gammabar = gammabar - gamma + dgamma
        
        tempv, templam, tempeta = LL(du, dgamma, mask, alp, bet)
        vbar = v + tau * tempv
        lambar = lam + tau * templam
        etabar = eta + tau * tempeta
        
        if False and samples != 0 and it % int(maxit/samples) == 0:
            dist = dist + [wass1_4d(u, toydata, M)]
            disteucl = disteucl + [np.linalg.norm(normalize_4d(u)[0] - normalize_4d(toydata)[0])]
        
        if prints != 0 and (it+1) % int(maxit/prints) == 0:
            out = out + [u]
            print(it+1)
            if info != None: info = (it+1, sig, tau, alp, bet, xi)
            plot_u(u, info=f'Wasserstein-TV regularized output after {it+1} iterations', save=saveall, name = name + str(it+1), plotall=plotall)
    
    return out, dist, disteucl

def CP(sig, tau, alp, bet, u, gamma, v, lam, eta, b, mask, maxit, prints=10):
    R1, R2, Q1, Q2 = np.shape(ubar)
    for it in range(maxit):
        tempu, tempgamma = LLstar(v, lam, eta, mask, alp, bet)
        u1 = np.real((u - sig* tempu)).clip(min=0)
        gamma1 = prox_norm21(sig, gamma - sig* tempgamma, ax=1)
        tempv, templam, tempeta = LL(2*u1-u, 2*gamma1-gamma, mask, alp, bet)
        v = v + tau* tempv
        lam = (lam + tau*templam - tau* b)/(1+tau)
        eta = (eta + tau* tempeta).clip(min=-1, max=1)
        u = u1.copy()
        gamma = gamma1.copy()
        if (it+1) % int(maxit/prints) == 0:
            print(it+1)
            plot_u(u, (it+1, sig, tau, alp, bet, xi), save=False)   
    return

def PG(u, b, mask, step, maxit, prints=1, samples=5): # perform GD without any TV-regularization
    out = []
    dist = []
    disteucl = []
    for it in range(maxit):
        u = u - step* KKstar(KK(u, mask) - b, mask)
        u = u.clip(min=0)
        if samples != 0 and it % int(maxit/samples) == 0:
            dist = dist + [wass1_4d(u, toydata, M)]
            disteucl = disteucl + [np.linalg.norm(normalize_4d(u)[0] - normalize_4d(toydata)[0])]
        if (it+1) % int(maxit/prints) == 0:
            out = out + [u]
            print(it+1)
            plot_u(u, info=f'Uregularized output after {it+1} iterations', save=False)
    return out, dist, disteucl

def PG_tik(u, b, mask, alpha, step, maxit, prints=1, samples=5):
    out = []
    dist = []
    disteucl = []
    for it in range(maxit):
        u = u - step* KKstar(KK(u, mask) - b, mask) - alpha*u
        u = u.clip(min=0)
        if samples != 0 and (it+1) % int(maxit/samples) == 0:
            dist = dist + [wass1_4d(u, toydata, M)]
            disteucl = disteucl + [np.linalg.norm(u - toydata)]
        if (it+1) % int(maxit/prints) == 0:
            out = out + [u]
            print(it+1)
            plot_u(u, info=f'Tikhonov output after {it+1} iterations', save=False)
    return out, dist, disteucl

def DR_standardtv(sig, tau, alp, xi, ubar, gammabar, vbar, lambar, du, dgamma, b, mask,
       maxit, prints=2, saveall=False, name = 'noname_' , plotall=False, info=None, samples = 5):
    
    out = []
    dist = []
    disteucl = []
    
    R1, R2, Q1, Q2 = np.shape(ubar)
    
    for it in range(maxit):
        
        u = ubar.clip(min=0)
        gamma = prox_norm1(sig, gammabar)
        
        v = vbar.copy()
        lam = (lambar - tau*b) / (1+tau) 
        
        tempu, tempgamma = LL1star(2*v-vbar, 2*lam-lambar, mask, alp)
        bu = 2 * u - ubar - sig * tempu
        bgamma = 2 * gamma - gammabar - sig * tempgamma
        
        tempbu, tempbgamma = T1(du, dgamma, sig, tau, mask, alp)
        tempdu, tempdgamma = M_inv(bu - tempbu, bgamma - tempbgamma, xi, precond='Richardson')
        du = du + tempdu 
        dgamma = dgamma + tempdgamma
        
        ubar = ubar - u + du
        gammabar = gammabar - gamma + dgamma
        
        tempv, templam = LL1(du, dgamma, mask, alp)
        vbar = v + tau * tempv
        lambar = lam + tau * templam
        
        if samples != 0 and it % int(maxit/samples) == 0:
            dist = dist + [wass1_4d(u, toydata, M)]
            disteucl = disteucl + [np.linalg.norm(normalize_4d(u)[0] - normalize_4d(toydata)[0])]
        
        if prints != 0 and (it+1) % int(maxit/prints) == 0:
            out = out + [u]
            print(it+1)
            if info != None: info = (it+1, sig, tau, alp, bet, xi)
            plot_u(u, info=f'Standard-TV regularized output after {it+1} iterations', save=saveall, name = name + str(it+1), plotall=plotall)
    
    return out, dist, disteucl

np.random.seed(42)

### UNDERSAMPLING MASK ###

#mask = generalized_undersampling_mask(datashape, fraction=0.2, scheme='gaussian', var=20, generalized=False)
mask_q = undersampling_mask_y(coord_y0, coord_y1, ratio=0.7, scheme='gaussian')
# mask = mask_kq(datashape, mask_q, ratio=0.7, scheme_k='random', variable_k=False)
# mask = np.ones(mask.shape, dtype=bool) # !! no mask imposed
mask = undersampling_mask(*[np.random.rand(k) for k in datashape], ratio=0.5, scheme='gaussian', cov=None, sym=False, generalized=False)

################################################

# plot_u(toydata)

noise_intensity = 0.02
b0 = KK(toydata, mask)
b = b0 + np.random.normal(0, noise_intensity, size=np.shape(b0))

if np.sum(mask) == np.size(toydata):
    plot_u(KKstar(b, mask), info='Plot of noisy image')

ubar = np.random.rand(*datashape)
# ubar = np.zeros(datashape)
ubar2=ubar.copy()
du = ubar.copy()

vbar = nabla_r(ubar)
lambar = KK(ubar, mask)
etabar = nabla_r(II(ubar))

gammabar = nabla2_q(vbar)
dgamma = gammabar.copy()

# M = buildcost() !! reactivate

maxit = 10000
samples = 20
prints_DR = 4
if_DR, if_PG, if_DR_tv = True, False, False

if if_DR:
    sig, tau, alp, bet = 0.5, 0.5, 0.01, 0.05
    tempxi = np.max([16, 64*alp**2 + 1 + 8*(alp*bet)**2*Q1*Q2])
    xi = 1 + sig*tau*tempxi
    ustars_DR, distances_DR, distances_eucl_DR = DR(sig, tau, alp, bet, xi, ubar, gammabar,
                                                    vbar, lambar, etabar, du, dgamma, b,
                                                    mask, maxit, prints=prints_DR, samples=samples)
    # ustar_DR = ustars_DR[-1]
    # dist_DR, disteucl_DR = distances_DR[-1], distances_eucl_DR[-1]

if if_PG:
    ustars_PG, distances_PG, distances_eucl_PG = PG(ubar2, b, mask, 0.5, maxit, prints=1,
                                                    samples=samples)
    ustar_PG = ustars_PG[-1]
    dist_PG, disteucl_PG = distances_PG[-1], distances_eucl_PG[-1]

if if_DR_tv:
    sig, tau, alp = 0.5, 0.05, 0.05
    tempxi = np.max([2, 64*alp**2 + 1])
    xi = 1 + sig*tau*tempxi
    ustars_DR_tv, distances_DR_tv, distances_eucl_DR_tv = DR_standardtv(sig, tau, alp, xi, ubar, vbar,
                                                    vbar, lambar, du, vbar, b,
                                                    mask, maxit, prints=prints_DR, samples=samples)
    ustar_DR_tv = ustars_DR_tv[-1]
    dist_DR_tv, disteucl_DR_tv = distances_DR_tv[-1], distances_eucl_DR_tv[-1]

if False:
    x = np.linspace(1,maxit-1,samples)
    if if_DR: plt.plot(x,distances_eucl_DR[1:])
    # if if_PG: plt.plot(x,distances_eucl_PG[1:])
    if if_DR_tv: plt.plot(x,distances_eucl_DR_tv[1:])
    plt.title('Euclidean distances')
    plt.show()
    
    if if_DR: plt.plot(x,distances_DR[1:])
    # if if_PG: plt.plot(x,distances_PG[1:])
    if if_DR_tv: plt.plot(x,distances_DR_tv[1:])
    plt.title('Wasserstein distances')
    plt.show()
    
    # print('Wass. dist. from data without/with regularization: ', round(dist_PG,2), '/', round(dist_DR,2) )
    # print('Euclidean dist. from data without/with regularization: ', round(disteucl_PG,2), '/', round(disteucl_DR,2))
    
# plot_u(img_wass, info='TV-Wasserstein regularized', save=True, name='img_wass.jpg')
# plot_u(img_wass_tv, info='TV regularized', save=True, name='img_wass_tv.jpg')

# x = np.linspace(1,maxit,len(distances_DR)-1)
# plt.plot(x,distances_eucl_DR[1:], label='TV-Wasserstein regularization')
# plt.plot(x,distances_eucl_DR_tv[1:], label='TV regularization')
# plt.legend()
# plt.title('Euclidean distances')
# plt.savefig('graph_euclidean_dist.jpg', dpi=200)
# plt.show()

# plt.plot(x,distances_DR[1:], label='TV-Wasserstein regularization')
# # if if_PG: plt.plot(x,distances_eucl_PG[1:])
# plt.plot(x,distances_DR_tv[1:], label='TV regularization')
# plt.legend()
# plt.title('Wasserstein distances')
# plt.savefig('graph_wasserstein_dist.jpg', dpi=200)
# plt.show()
    
'''
    
plot_u(u, info=None, save=False, toplot=(4,4), plotall=False, name='newname.jpg',
           title=None, vmin=None, vmax=None, plotkq=False):
    ustars_DR_2 = ustars_DR.copy()
    distances_DR_2 = distances_DR.copy()
    distances_eucl_DR_2 = distances_eucl_DR.copy()

    ustars_PG_2 = ustars_PG.copy()
    distances_PG_2 = distances_PG.copy()
    distances_eucl_PG_2 = distances_eucl_PG.copy()
    '''