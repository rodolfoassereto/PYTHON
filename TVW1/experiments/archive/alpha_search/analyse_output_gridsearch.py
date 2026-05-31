import numpy as np
import matplotlib.pyplot as plt

### Import the data from the gridsearch

output_parallel = np.load("output_parallel.npy", allow_pickle=True)
output_parallel = np.load("output_parallel_10alphas_30epsilons.npy", allow_pickle=True)
output_parallel = output_parallel.item()

def get_color(i, n, colormap='viridis'):
    ''' This function returns the color of the i-th graph (out of n) '''
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(0, n)
    return cmap(norm(i))

### Now we plot the various distances as a function of the regularization parameters

from collections import defaultdict

for which_distance in [ 'dist_W1', 'dist_W2', 'dist_L1', 'dist_L2' ]:

    dist = output_parallel[which_distance][(0.2, 0.01)]
    
    dist_alphakeys = defaultdict(dict)
    dist_epskeys = defaultdict(dict)
    
    all_alphas, all_epsilons = [], []
    for (alpha, eps), d in dist.items():
        dist_alphakeys[alpha][eps] = d
        dist_epskeys[eps][alpha] = d
        all_alphas += [alpha]
        all_epsilons += [eps]
    
    n_alphas, n_epsilons = len(dist_alphakeys.keys()), len(dist_epskeys.keys())
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for i, (alpha, eps_dict) in enumerate(dist_alphakeys.items()):
        
        color_alpha = get_color(i, n_alphas)
        
        epsilons = list(eps_dict.keys())
        distances = [eps_dict[eps] for eps in eps_dict.keys()]
        axes[0].plot(epsilons, distances, label=f'alpha = {np.round(alpha, decimals=3)}', color=color_alpha)
        axes[0].set_xlabel('beta')
        axes[0].grid(True)
        axes[0].legend(fontsize='x-small', loc='upper left', bbox_to_anchor=(1.05, 1))
    
    for i, (eps, alpha_dict) in enumerate(dist_epskeys.items()):
        
        color_eps = get_color(i, n_epsilons)
        
        alphas = sorted(alpha_dict.keys())
        distances = [alpha_dict[alpha] for alpha in alpha_dict.keys()]
        axes[1].plot(alphas, distances, label=f"beta = {eps:.2e}", color=color_eps)
        axes[1].set_xlabel('alpha')
        axes[1].grid(True)
        axes[1].legend(fontsize='x-small', loc='upper left', bbox_to_anchor=(1.05, 1))
    
    axes[0].set_ylabel('distance')
    fig.suptitle(which_distance)
    plt.tight_layout()
    plt.show()


### 

