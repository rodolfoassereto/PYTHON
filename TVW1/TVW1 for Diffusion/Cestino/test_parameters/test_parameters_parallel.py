import numpy as np

from itertools import product
from joblib import Parallel, delayed

from test_parameters import u_deg, keptvalues, noiselevels, alphas, epsilons, maxit
from test_parameters import undersampling_mask_small, UF, L2_TVPR_Dirichlet

from objective_functions import wass1_4d, wass2_4d, build_cost_matrix

from tqdm import tqdm # used to print the prgress in parallelised loops

M = build_cost_matrix(u_deg[0,0].shape) # cost matrix for dist_W1 and dist_W2

all_combinations = product(keptvalues, noiselevels, alphas, epsilons)
length_combinations = len(keptvalues) * len(noiselevels) * len(alphas) * len(epsilons)

def single_run(kept_val, noiselevel, alpha, eps, u_deg, maxit, i, M=M): 
    
    np.random.seed(0)
    mask = np.zeros(u_deg.shape, dtype=bool)
    mask[:,:] = undersampling_mask_small(u_deg.shape[-2:], ratio=kept_val)
    mask = np.invert(np.fft.ifftshift(mask))

    b_clean = UF(u_deg, mask=mask)
    b = b_clean + np.random.normal(size=u_deg.shape, scale=noiselevel * np.max(np.abs(b_clean)))

    u_star, _ = L2_TVPR_Dirichlet(b, alpha1=alpha, eps=eps, maxit=maxit, printprogress=False)

    dist_W1 = wass1_4d(u_star, u_deg, M=M)
    dist_W2 = wass2_4d(u_star, u_deg)
    dist_L1 = np.sum(np.abs(u_star - u_deg))
    dist_L2 = np.linalg.norm(u_star - u_deg)

    return (kept_val, noiselevel, alpha, eps), (dist_W1, dist_W2, dist_L1, dist_L2)

results = Parallel(n_jobs=-1)(
    delayed(single_run)(kept_val, noiselevel, alpha, eps, u_deg, maxit, i)
    for i, (kept_val, noiselevel, alpha, eps) in enumerate(tqdm(all_combinations, total=length_combinations))
)

dist_W1 = {}
dist_W2 = {}
dist_L1 = {}
dist_L2 = {}

for params, distances in results:
    kept_val, noiselevel, alpha, eps = params
    key_outer = (kept_val, noiselevel)
    key_inner = (alpha, eps)

    if key_outer not in dist_W1:
        dist_W1[key_outer] = {}
        dist_W2[key_outer] = {}
        dist_L1[key_outer] = {}
        dist_L2[key_outer] = {}

    dist_W1[key_outer][key_inner] = distances[0]
    dist_W2[key_outer][key_inner] = distances[1]
    dist_L1[key_outer][key_inner] = distances[2]
    dist_L2[key_outer][key_inner] = distances[3]


best_par_W1 = { par0: min(dist_W1[par0], key=dist_W1[par0].get)  for par0 in dist_W1.keys() }
best_par_W2 = { par0: min(dist_W2[par0], key=dist_W2[par0].get)  for par0 in dist_W2.keys() }
best_par_L1 = { par0: min(dist_L1[par0], key=dist_L1[par0].get)  for par0 in dist_L1.keys() }
best_par_L2 = { par0: min(dist_L2[par0], key=dist_L2[par0].get)  for par0 in dist_L2.keys() }

