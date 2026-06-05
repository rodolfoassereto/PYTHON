import sys, os
root_folder = "C:/Users/rodol/My Drive/PYHTON"
for root, dirs, files in os.walk(root_folder):
    sys.path.append(root)

import numpy as np
import matplotlib.pyplot as plt
from build_toymodels import build_u_gammas, generate_simple_gaussian_crossing, build_sebastian_distributions
from build_toymodels import generate_field, build_u_gaussians_fromfield, build_u_gaussian_mixture
# from objective_functions import wass1_4d, wass2_4d, build_cost_matrix
from plottings import plot_u

########### 1. BUILDING THE CLEAN DATA

'''
We build three types of data to be reconstructed with shape (M,M,N,N): The parameters are:
    M, N: the original data is supposed to have shape (M,M,N,N)
'''

ndim_y = 2

########### 1.1) build u gammas

M_1, N_1 = 5, 30
u_1 = build_u_gammas(M=M_1, N=N_1) # One can plot 4-d data using plot_u
# plot_u(u_1)

########### 1.2) build u gaussians

# M_2 = 9
# u_2 = generate_simple_gaussian_crossing(M_2)
# u_gaussians = build_u_gaussians_fromfield( generate_field(M_gaussians,1,0,0) , generate_field(M_gaussians,0,0.5,-1) )
from build_toymodels import flow_gaussians_crossing as u_2
u_2 = u_2[2:-2,2:-2]
plot_u(u_2, ndim_y)

########### 1.3) build u brain (from tensors)
'''
This data is build upon the tensors from Sebastian.
We are considering only a piece of a slice, and we are considering only planar displacements.
Ultimately, however, a full array will need to be considered.
'''
N_3=11
u_3, mask_u3 = build_sebastian_distributions(N=N_3) # slice of gaussians from Sebastian's tensors
u_3 = u_3[:,:,int(N_3/2)]
# u_3 = u_3_full[20:60,20:60]
# plot_u(u_3)

########### 2. ASSESSING REFERENCE VALUES FOR UNDERSAMPLING AND NOISE
'''
To build the measured data 'b', we import the forward operator KK (and its adjoint KKstar), which
performs an undersampled Fourier transform.
The involved parameters for the construction of b are:
    keptval:    percentage of sampled values in the Fourier space
    noiselevel: percentage of noise w.r.t. np.max( np.abs( KK(u) ) )
'''

from build_toymodels import undersampling_mask_small, KK, KKstar

def build_b(data, keptval, noiselevel):
    np.random.seed(0)
    mask = np.zeros(data.shape, dtype=bool)
    mask[:,:] = undersampling_mask_small(data.shape[-2:], ratio=keptval)
    mask = np.invert( np.fft.ifftshift(mask) )
    b_clean = KK(data, mask=mask)
    b = b_clean + np.random.normal(size=data.shape, scale=noiselevel * np.max(np.abs(b_clean)))
    return b, b_clean

########### 2.1) build b from gammas with suitable parameters

keptval_1 = 0.15
noiselevel_1 = 0.02 # 2% of np.max( np.abs( KK(u) ) )

b_1, b_1_clean = build_b(u_1, keptval_1, noiselevel_1)
u_1_rough = np.real( KKstar( b_1 ) ) # this is a rough reconstruction, only involving the adjoint operator
# plot_u(u_1_rough)

########### 2.2) build b from gaussians with suitable parameters

keptval_2 = 0.155
noiselevel_2 = 0.0075

b_2, b_2_clean  = build_b(u_2, keptval_2, noiselevel_2)
u_2_rough = np.real( KKstar( b_2 ) )

plot_u(u_2_rough, ndim_y)

########### 2.3) build b from brain with suitable parameters

keptval_3 = 0.2
noiselevel_3 = 0.01

b_3, b_3_clean = build_b(u_3, keptval_3, noiselevel_3)
u_3_rough = np.real( KKstar( b_3 ) )
# plot_u(u_rough)


########### 3. ASSESSING REFERENCE VALUES FOR RECONSTRUCTION PARAMETERS

'''
We'll be testing the reconstructions with the model L2_TVPR, with parameters:
    alpha1: intensity of the Optimal Transport-based regularizer (TVPR)
    eps:    intensity of the smoothness regularizer (TV_y) (imposes smoothness in each voxel)
    maxit:  number of iterations
The signature of the function is:
    def L2_TVPR(b, alpha1, eps, beta1 = None, forward='KK', maxit=10000):
        ...
        return u, w
    where u is the reconstructed variable.
We also import L2_TVPR_Dirichlet, which has a different smoothing regularizer (Dirichlet energy instead of TV_y)
We expect it to be more effective on 'continuous' images, such as gaussians, rather than cartoon-like, such as gammas.
Here, we heuristically pinpoint some reference parameters on which we'll center the grids in our search
'''

from models_2 import L2_TVPR, L2_TVPR_Dirichlet

########### 3.1) gammas reconstruction

alpha1_1 = 0.01
eps_1 = 0.005
# u_1_star_0, _ = L2_TVPR(b_1, alpha1_1, eps_1, maxit=200, printprogress=True)
# plot_u(u_1_star_0)

########### 3.2) gaussians reconstruction

alpha1_2 = 0.002
eps_2 = 0

########### 3.3) brain reconstruction

alpha1_3 = 0.05
eps_3 = 0

########### 4. RECONSTRUCTION AND EVALUATION THROUGH GRIDSEARCH

'''
We can employ different evaluation methods for our solution:
    1) Using a Wasserstein-1 (W1) distance
    2) Using a Wasserstein-2 (W2) distance
    3) Using a Euclidean (L2) or L1 distance
Note: W2 might be the most reliable indicator.
Since we only care about the shape of the reconstructions (and not the intensities within each voxel), we can normalize each voxel
'''

########### 4.1) gammas

# For now, we only run the search for the gaussians

########### 4.2) gaussians

u_deg = u_2.copy()
b = b_2.copy()

keptval = keptval_2
noiselevel = noiselevel_2
alpha1 = alpha1_2
eps = eps_2

########### 4.3) brain

# u_deg = u_3.copy()

# keptval = keptval_3
# noiselevel = noiselevel_3
# alpha1 = alpha1_3
# eps = eps_3

########### 4.4) gridsearch

keptvalues = (keptval,)
# noiselevels = (noiselevel,)
noiselevels = (0, 0.0005, 0.001, 0.002, 0.004, 0.007, 0.01)

base = 1.2
exp = np.emath.logn(base, alpha1)
alphas = base ** np.linspace( exp-12, exp+7, 20 )

epsilons = (0,) # for now, we keep epsilon=0

maxit = 1000

# u_deg is supposed to be the clean u
def gridsearch(u_deg=u_deg, keptvalues=keptvalues, noiselevels=noiselevels, alphas=alphas, epsilons=epsilons, maxit=maxit):
    
    l0, l1, l2, l3 = len(keptvalues), len(noiselevels), len(alphas), len(epsilons)
    num_par = np.prod( [l0, l1, l2, l3] )
    
    import time
    start_time = time.time()
    next_update = start_time
    
    dist_W1, dist_W2, dist_L1, dist_L2 = {}, {}, {}, {}
    dist_W1_rough, dist_W2_rough, dist_L1_rough, dist_L2_rough = {}, {}, {}, {}
    
    M = build_cost_matrix(u_deg[0,0].shape)
    
    for i0, kept_val in enumerate(keptvalues):
        
        np.random.seed(0)
        
        mask = np.zeros(u_deg.shape, dtype=bool)
        mask[:,:] = undersampling_mask_small(u_deg.shape[-2:], ratio=kept_val)
        mask = np.invert( np.fft.ifftshift(mask) )
        
        for i1, noiselevel in enumerate(noiselevels):
            
            b_clean = KK(u_deg, mask=mask)
            b = b_clean + np.random.normal(size=u_deg.shape, scale=noiselevel * np.max(np.abs(b_clean)))
            
            u_rough = np.abs( KKstar(b) )
            # plot_u(u_rough)
            dist_W1_rough[ (kept_val, noiselevel) ] = wass1_4d( u_rough, u_deg, M=M)
            dist_W2_rough[ (kept_val, noiselevel) ] = wass2_4d( u_rough, u_deg )
            dist_L1_rough[ (kept_val, noiselevel) ] = np.max(np.abs( u_rough - u_deg ) )
            dist_L2_rough[ (kept_val, noiselevel) ] = np.linalg.norm( u_rough - u_deg )
            
            dist_W1[ (kept_val, noiselevel) ] = {}
            dist_W2[ (kept_val, noiselevel) ] = {}
            dist_L1[ (kept_val, noiselevel) ] = {}
            dist_L2[ (kept_val, noiselevel) ] = {}
        
            for i2, alpha in enumerate(alphas):
                
                for i3, eps in enumerate(epsilons):
                    
                    current_time = time.time()
                    if (current_time >= next_update):
                        print(f"Progress: {i0*(l1+l2+l3)+i1*(l2+l3)+i2*l3+i3}/{num_par}")
                        next_update = current_time + 180  # Schedule next update
                    u_star, _ = L2_TVPR_Dirichlet(b, alpha1=alpha, eps=eps, maxit=maxit, printprogress=False)
                    
                    dist_W1[ (kept_val, noiselevel) ][ (alpha, eps) ] = wass1_4d(u_star, u_deg, M=M)
                    dist_W2[ (kept_val, noiselevel) ][ (alpha, eps) ] = wass2_4d(u_star, u_deg)
                    dist_L1[ (kept_val, noiselevel) ][ (alpha, eps) ] = np.sum( np.abs(u_star - u_deg) )
                    dist_L2[ (kept_val, noiselevel) ][ (alpha, eps) ] = np.linalg.norm(u_star - u_deg)
    
    distances_rough = dist_W1_rough, dist_W2_rough, dist_L1_rough, dist_L2_rough
    distances = dist_W1, dist_W2, dist_L1, dist_L2
    
    return distances, distances_rough
    
''' This code launches the gridsearch function and might have been commented out in order for the script
to compile fast '''

# distances, distances_rough = gridsearch()

# dist_W1_rough, dist_W2_rough, dist_L1_rough, dist_L2_rough = distances_rough
# dist_W1, dist_W2, dist_L1, dist_L2 = distances

# best_par_W1 = { par0: min(dist_W1[par0], key=dist_W1[par0].get)  for par0 in dist_W1.keys() }
# best_par_W2 = { par0: min(dist_W2[par0], key=dist_W2[par0].get)  for par0 in dist_W2.keys() }
# best_par_L1 = { par0: min(dist_L1[par0], key=dist_L1[par0].get)  for par0 in dist_L1.keys() }
# best_par_L2 = { par0: min(dist_L2[par0], key=dist_L2[par0].get)  for par0 in dist_L2.keys() }

# kept, noise = next(iter(best_par_W2))
# alpha, eps = best_par_W2[ (kept, noise) ]

# np.random.seed(0)

# mask = np.zeros(u_deg.shape, dtype=bool)
# mask[:,:] = undersampling_mask_small(u_deg.shape[-2:], ratio=kept)
# mask = np.invert( np.fft.ifftshift(mask) )

# b_clean = KK(u_deg, mask=mask)
# b = b_clean + np.random.normal(size=u_deg.shape, scale=noise * np.max(np.abs(b_clean)))

# plot_u( np.real( KKstar(b) ) )

# u_star, _ = L2_TVPR(b, alpha1=alpha, eps=eps, maxit=7500, printprogress=True)

# plot_u(u_star)


# # ######### 5. IDEAS FOR POSSIBLE IMPROVEMENTS (not relevant atm)

# # from objective_functions import W22
# # from plottings import printcurve

# # f = np.zeros((11,11))
# # g = f.copy()

# # f[1,1] = 1
# # f[-2,1] = 1
# # g[5,-1] = 2

# # plt.imshow(f, cmap='gray')
# # plt.show()
# # plt.imshow(g, cmap='gray')
# # plt.show()

# # X1, Y1 = W22(f, g, M_grid=20, maxit=25000)

# # printcurve(X1)


