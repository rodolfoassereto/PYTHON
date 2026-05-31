# %% #################### Imports

import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import sys
# Add Libraries folder to sys.path
libraries_path = r"C:\Users\rodol\My Drive\PYHTON\Libraries"
if libraries_path not in sys.path:
    sys.path.insert(0, libraries_path)
from plottings import plot_u  # noqa: E402


def generate_new_grid( center, old_grid, zoom_factor=0.5 ): # pensato per una griglia di lunghezza 4
    
    a, b = np.min(old_grid), np.max(old_grid)
    old_radius = (b - a) / 2
    num_gridpoints = len(old_grid)
    
    new_radius = zoom_factor * old_radius
    
    if center <= a:
        a1 = center - old_radius
        b1 = center + new_radius
    
    elif center >= b:
        a1 = center - new_radius
        b1 = center + old_radius
    
    else:
        a1 = center - new_radius
        b1 = center + new_radius
    
    return np.linspace( np.max((a1,0)), b1, num_gridpoints)

# %% #################### Experiment 1: stability of best alpha vs. SNR

from test_models_with_gaussian_mixture import undersampling_mask_xy, shape_x, shape_y, gaussian_mixture, KK, KKstar  # noqa
from L2_TVW1_basic__model import model_basic  # noqa

shape_xy = shape_x + shape_y
ndim_x, ndim_y = len(shape_x), len(shape_y)


SNR = 20
retained_ratio = 0.125

distances_dict = {}

N = 8

for k in range(N):
    
    print(k+1,'/',N)
    
    distances = {}

    mask_fft = undersampling_mask_xy(shape_x, shape_y, retained_ratio, concentration_coeff=0.7)
    plot_u(mask_fft, ndim_y)
    mask_fft = np.fft.ifftshift(mask_fft)
    mask_rfft = mask_fft[..., : (shape_xy[-1]//2 + 1)]
    
    E_clean = KK( gaussian_mixture, mask_rfft)
    size_E = E_clean.size
    E_energy = np.sum(np.abs(E_clean)**2)
    
    standard_deviation = np.sqrt( E_energy / ( size_E * SNR ) )
    noise = np.random.normal( size=size_E, scale=standard_deviation ) +1j* np.random.normal( size=size_E, scale=standard_deviation )
    
    E = E_clean + noise/np.sqrt(2)
    
    # plot_u( KKstar(E, mask_rfft).clip(min=0), ndim_y )
    
    iterations_short = 1000
    stepsize_ratio = 0.5
    
    beta = 0
    
    distances = {}
    
    grid_alphas = np.logspace(-3,0,10)
    for i, alpha in enumerate(grid_alphas):
        # print(i,'/',len(grid_alphas)-1)
        Pstar = model_basic(E, mask_rfft, shape_x, shape_y, alpha, beta, stepsize_ratio, iterations_short, printprogress=False)
        dist = np.sum((Pstar - gaussian_mixture)**2)
        distances[alpha] = dist
    
    distances_dict[k] = distances


fig, ax = plt.subplots()
for k in (1,2,3,4,5,6,7):
    ax.plot(grid_alphas, distances_dict[k].values())
    ax.set_title(f'dist vs. alpha; experiments with {N} different masks ({SNR}/{retained_ratio})')
    ax.set_xlabel('alpha')
    ax.set_ylabel('dist')
plt.show()


# %% #################### Experiment 2: alpha vs (SNR, retained_ratio) 

SNRs = np.array( [10, 12, 15, 20, 40, 80] )                                     # NOISE PARAMETERS
retained_ratios = np.array( [ 0.10, 0.12, 0.15, 0.2, 0.3, 0.7 ] )

number_of_bisections = 6
zoom_factor = 0.33
num_gridpoints = 4
grid_alphas_start = np.linspace(0, 0.6, num_gridpoints)

distances_dict = {}
alphas_dict = {}
Pstars_dict = {}
Es_dict = {}

for counter, retained_ratio in enumerate(retained_ratios):
    
    print( counter+1, '/', len(retained_ratios) )
    
    mask = undersampling_mask_xy(shape_x, shape_y, retained_ratio, 0.7)
    mask = np.fft.ifftshift(mask)
    mask_rfft = mask[..., : (shape_xy[-1]//2 + 1)]
    
    E_clean = KK( gaussian_mixture, mask_rfft )
    
    for SNR in SNRs:
        
        E_energy = np.sum( np.abs(E_clean)**2 )
        size_E = E_clean.size
        standard_deviation = np.sqrt( E_energy / ( size_E * SNR ) )
        noise = np.random.normal( size = size_E, scale = standard_deviation ) + 1j* np.random.normal( size=size_E, scale=standard_deviation )
        
        E = E_clean + noise / np.sqrt(2)
        
        Es_dict[(retained_ratio, SNR)] = E
        
        iterations_short = 750
        stepsize_ratio = 0.5
        
        beta = 0
        
        distances_temp = {}
        best_distance_temp = np.inf
        
        grid_alphas = grid_alphas_start
        
        for _ in range(number_of_bisections):            
        
            for alpha in grid_alphas:
                
                if distances_temp.get(alpha) is None: # controllo se ho già calcolato questo alpha
                    
                    Pstar = model_basic(E, mask_rfft, shape_x, shape_y, alpha, beta, stepsize_ratio, iterations_short, printprogress=False)
                    dist = np.linalg.norm( Pstar - gaussian_mixture )
                    if dist < best_distance_temp:
                        best_distance_temp = dist
                        Pstars_dict[(retained_ratio, SNR)] = Pstar
                    distances_temp[alpha] = dist
                
            center = min(distances_temp, key=distances_temp.get) # seleziono l'argmin di distances_temp (che dovrebbe essere best_distance_temp)
            
            grid_alphas = generate_new_grid(center, grid_alphas, zoom_factor=zoom_factor)   
        
        alphas_dict[(retained_ratio, SNR)] = center
        distances_dict[(retained_ratio, SNR)] = distances_temp[center]

# %%

# sorted 1D arrays
snr_vals   = np.sort(SNRs)[1:]
ratio_vals = np.sort(retained_ratios)[1:]

# build 2D array of best alphas
alpha_grid = np.zeros((len(snr_vals), len(ratio_vals)))
for i, snr in enumerate(snr_vals):
    for j, rr in enumerate(ratio_vals):
        alpha_grid[i, j] = alphas_dict[(rr, snr)]

fig, ax = plt.subplots(figsize=(7, 4))

# no extent: indices 0..n-1 on each axis
im = ax.imshow(alpha_grid, origin='lower', aspect='auto')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"best $\alpha$")

# put one tick per column/row and label with the actual values
ax.set_xticks(np.arange(len(ratio_vals)))
ax.set_xticklabels([f"{r:.2f}" for r in ratio_vals])
ax.set_yticks(np.arange(len(snr_vals)))
ax.set_yticklabels([str(s) for s in snr_vals])

ax.set_xlabel("retained_ratio")
ax.set_ylabel("SNR")
ax.set_title(r"Best $\alpha$ over (retained\_ratio, SNR)")

plt.tight_layout()
plt.show()


# To plot the Pstars:

# for key in Pstars_dict:
#     u = Pstars_dict[key]
#     retained_ratio, SNR = key
#     plot_u(u, 2, title=f"retained_ratio={retained_ratio}, SNR={SNR}")


# %% #################### Experiment 3: how does alpha change with size?

from test_models_with_gaussian_mixture import build_gaussian_mixture  # noqa: E402

scaling_factor_x = 1.5
scaling_factor_y = 1.5
num_scales_x = 4
num_scales_y = 4

sh_x_start = (7, 7)
sh_y_start = (6, 6)

shapes_x = [ tuple( np.floor( scaling_factor_x**k * np.array(sh_x_start) ).astype(int) ) for k in range(num_scales_x) ]
shapes_y = [ tuple( np.floor( scaling_factor_y**k * np.array(sh_y_start) ).astype(int) ) for k in range(num_scales_y) ]

# Set a retained_ratio and a SNR
retained_ratio = 0.2
SNR = 20

number_of_bisections = 6
zoom_factor = 0.33
num_gridpoints = 4
grid_alphas_start = np.linspace(0, 0.6, num_gridpoints)

iterations_short = 1250
stepsize_ratio = 0.5

beta = 0

gaussian_mixtures_dict = {}
Pstars_dict = {}
alphas_dict = {}

for sh_x, sh_y in it.product(shapes_x, shapes_y):
    
    ndim_x_loop = len(sh_x)
    ndim_y_loop = len(sh_y)
    sh_xy = sh_x + sh_y
    ndim_xy = len(sh_xy)
    
    print(f"Current shape: {sh_xy}")
    
    gaussian_mixture = build_gaussian_mixture(sh_x, sh_y)
    gaussian_mixture = 100 * gaussian_mixture / np.sum(gaussian_mixture, axis=tuple(np.arange(ndim_x_loop, ndim_xy))).reshape(sh_x + ndim_y_loop*(1,))
    
    mask = undersampling_mask_xy(sh_x, sh_y, retained_ratio, concentration_coeff=0.7)
    mask = np.fft.ifftshift(mask)
    mask_rfft = mask[..., : (sh_xy[-1]//2 + 1)]

    gaussian_mixtures_dict[(sh_x, sh_y)] = gaussian_mixture
    # plot_u(gaussian_mixture, ndim_y_loop, title=f"Current shape: {sh_xy}")
    
    E_clean = KK(gaussian_mixture, mask_rfft)
    E_energy = np.sum( np.abs(E_clean)**2 )
    size_E = E_clean.size
    standard_deviation = np.sqrt( E_energy / ( size_E * SNR ) )
    noise = np.random.normal( size = size_E, scale = standard_deviation ) + 1j* np.random.normal( size=size_E, scale=standard_deviation )
    E = E_clean + noise / np.sqrt(2)
    
    distances_temp = {}
    best_distance_temp = np.inf
    
    grid_alphas = grid_alphas_start
    for _ in range(number_of_bisections):
        for alpha in grid_alphas:
            
            if distances_temp.get(alpha) is None: # controllo se ho già calcolato questo alpha
                
                Pstar = model_basic(E, mask_rfft, sh_x, sh_y, alpha, beta, stepsize_ratio, iterations_short, printprogress=False)
                dist = np.linalg.norm( Pstar - gaussian_mixture )
                distances_temp[alpha] = dist
                if dist < best_distance_temp:
                    best_distance_temp = dist
                    Pstars_dict[(sh_x, sh_y)] = Pstar
            
        center = min(distances_temp, key=distances_temp.get) # seleziono l'argmin di distances_temp (che dovrebbe essere best_distance_temp)
        
        grid_alphas = generate_new_grid(center, grid_alphas, zoom_factor=zoom_factor)   
    
    alphas_dict[(sh_x, sh_y)] = center

# Plots

nx_list   = []
ny_list   = []
ntot_list = []
alpha_list = []

for (sx, sy), alpha in alphas_dict.items():
    nx = np.prod(sx)   # spatial voxels
    ny = np.prod(sy)   # angular samples per voxel
    nx_list.append(nx)
    ny_list.append(ny)
    ntot_list.append(nx * ny)
    alpha_list.append(alpha)

nx   = np.array(nx_list)
ny   = np.array(ny_list)
ntot = np.array(ntot_list)
alpha = np.array(alpha_list)


plt.figure()
plt.loglog(ntot, alpha, 'o')
plt.xlabel("total unknowns n_x * n_y")
plt.ylabel("best alpha")
plt.title("Best alpha vs total size (log-log)")
plt.grid(True, which='both', ls=':')
plt.show()

# group by unique ny values
unique_ny = np.unique(ny)

plt.figure()
for ny_val in unique_ny[1:]:
    mask = (ny == ny_val)
    plt.plot(nx[mask], alpha[mask], 'o-', label=f"ny={ny_val}")
plt.xlabel("n_x")
plt.ylabel("best alpha")
plt.legend()
plt.grid(True, which='both', ls=':')
plt.title("alpha vs n_x for fixed n_y")
plt.show()

# group by unique nx values

unique_nx = np.unique(nx)

plt.figure()
for nx_val in unique_nx:
    mask = (nx == nx_val)
    plt.plot(ny[mask], alpha[mask], 'o-', label=f"n_x={nx_val}")

plt.xlabel("n_y")
plt.ylabel("best alpha")
plt.legend(title="fixed n_x")
plt.grid(True, which='both', ls=':')
plt.title(r"$\alpha$ vs $n_y$ for fixed $n_x$")
plt.show()

# %% #################### [...] Experiment 4: level sets of Pstar vs (alpha, beta)
