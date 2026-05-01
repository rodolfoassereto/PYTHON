'''
This code runs the algorthm.
The file image is 'Woman.jpg'.
- "biggest_size" is the size of the finer grid (side = 2**biggest_size). It should not exceed the size of the file image.
- "k" is the number of smaller grids (so that the total of grids is k+1)
- "alpha, sigma1, sigma2, lam" are the parameters for the algorithm
- "F_obj" + "Gstar_obj" = "objective_function" (the objective funcion from the primal problem)
- 

'''

import numpy as np
import matplotlib.pyplot as plt
import gradients_and_TV_multigrid as tv
import prox_FF as pf
import prox_GG as pg
import building_data as bd
import multiscale_DR as DR

import sys
sys.path.append('C:\\Users\\rodol\\Google Drive\\PYHTON\\Multigrid Denoising Spyder')

biggest_size = 8 # With the file 'Woman.jpg' the biggest possible size is 9
k = 4 # number of sublevels
noiselevel = 0.2

alpha, sigma1, sigma2 = 5, 2, 2
lam = np.max([1 + 9*sigma1*sigma2, 1+sigma1*sigma2*(k+1)])+1

# Here we build the data for the initial guess

im_original = bd.import_image('Woman.jpg', biggest_size)
im_noisy = bd.add_gaussian_noise(im_original, sig=noiselevel)
zz = bd.build_list_of_images(im_noisy, k)
uu0 = np.array([z.copy() for z in zz], dtype=object)
r0, s0 = 1, 1
vv0 = np.array([z.copy() for z in zz], dtype=object) 
ww0 = np.empty(len(vv0), dtype=object)
ww0[:] = [ tv.nabla(x) for x in vv0 ]
tt0 = np.array((k+1)*[1])
ttau0 = tt0.copy()
duu0 = np.array([z.copy() for z in zz], dtype=object)
dr0, ds0 = 1, 1
baruu0 = np.array([z.copy() for z in zz], dtype=object)
barr0, bars0 = 1, 1
barvv0 = np.array([z.copy() for z in zz], dtype=object)
barww0 = np.empty(len(vv0), dtype=object)
barww0[:] = [ tv.nabla(x) for x in vv0 ]
bartt0 = tt0.copy()
barttau0 = ttau0.copy()

# plt.imshow(im_original, cmap='gray')
# plt.show()

def F_obj(uu, r, s):
    uuu = pf.SS(uu)
    for u in uuu:
        if np.linalg.norm(u) > 1e-3: return np.inf
    return r + s

def Gstar_obj(vv, ww, tt, ttau):
    for i in range(len(vv)):
        if pg.g(i, vv[i], zz[i]) > tt[i] + 1e-3: return np.inf
        if pg.h(alpha, i, ww[i]) > ttau[i] + 1e-3: return np.inf
    return 0

def objective_function(uu, r, s):
    vv, ww, tt, ttau = DR.KK(uu, r, s)
    return F_obj(uu,r,s) + Gstar_obj(vv,ww,tt,ttau)

maxit = 1000

for i in range(maxit):
    if i%10 == 0:
        plt.imshow(uu0[0], vmin=0, vmax=1, cmap='gray')
        plt.title('(k = {}) Iteration number {}/{}'.format(k, i+1, maxit))
        plt.show()        
    #print('TV of the bigger image at step ', i,': ',tv.TV(uu0[0]))
    uu0, r0, s0, vv0, ww0, tt0, ttau0, duu0, dr0, ds0, baruu0, barr0, bars0, barvv0, barww0, bartt0, barttau0 = DR.multiscale_pDR_step(alpha, uu0, r0, s0, vv0, ww0, tt0, ttau0, duu0, dr0, ds0, baruu0, barr0, bars0, barvv0, barww0, bartt0, barttau0, sigma1, sigma2, zz, lam)

plt.imshow(uu0[0], vmin=0, vmax=1, cmap='gray')
plt.title('(k = {}) Iteration number {}/{}'.format(k, i+1, maxit))
plt.show()
plt.imshow(im_noisy, cmap='gray')
plt.show()

plt.imshow(np.zeros((1,1)), cmap='gray')
plt.show()

print('(k = {}) Final TV = {}'.format(k, tv.TV(uu0[0])))