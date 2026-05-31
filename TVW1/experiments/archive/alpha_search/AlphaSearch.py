# %%

import numpy as np
import time
import itertools as it
import matplotlib.pyplot as plt

from L2_TVW1_naif__model import model
from test_models_with_gaussian_mixture import gaussian_mixture, shape_x, shape_y, mask_rfft, E, alpha1, alpha2, stepsize_ratio

base1 = 1.25
exp1 = float(np.emath.logn(base1, alpha1))
range_alpha1 = tuple( base1 ** np.linspace( exp1-10, exp1+10, 21 ) )

base2 = 1.25
exp2 = float(np.emath.logn(base2, alpha2))
range_alpha2 = tuple( base2 ** np.linspace(exp2-8, exp2+8, 17) )

from objective_functions import build_cost_matrix, wass1_4d, wass2_4d
M = build_cost_matrix( shape_y )

dist_L2 = {}
dist_W1 = {}
dist_W2 = {}
Pstar_dict = {}

# %%

iterations = 1000

counter = 0
t0 = time.time()
for alpha1, alpha2 in it.product( range_alpha1, range_alpha2 ):

    Pstar_dict[(alpha1, alpha2)] = model( E, mask_rfft, shape_x, shape_y, alpha1, alpha2, stepsize_ratio, iterations, printprogress=False )

    dist_L2[(alpha1, alpha2)] = np.sum( ( Pstar_dict[(alpha1, alpha2)] - gaussian_mixture )**2 )
    dist_W1[(alpha1, alpha2)] = wass1_4d( Pstar_dict[(alpha1, alpha2)], gaussian_mixture, M=M )
    dist_W2[(alpha1, alpha2)] = wass2_4d( Pstar_dict[(alpha1, alpha2)], gaussian_mixture, M=M**2 )
    t1 = time.time()
    if t1 - t0 >= 30:
        print(round(counter*100 / (len(range_alpha1)*len(range_alpha2)), 1), "%  (", counter, "/", (len(range_alpha1)*len(range_alpha2)), ")")
        t0 = t1
    counter += 1

X_alphas, Y_alphas = np.meshgrid(range_alpha1, range_alpha2, indexing='ij')
dL2 = np.zeros(X_alphas.shape)
dW1 = np.zeros(X_alphas.shape)
dW2 = np.zeros(X_alphas.shape)
for idx in np.ndindex(*dL2.shape):
    dL2[idx] = dist_L2[ ( X_alphas[idx], Y_alphas[idx] ) ]
    dW1[idx] = dist_W1[ ( X_alphas[idx], Y_alphas[idx] ) ]
    dW2[idx] = dist_W2[ ( X_alphas[idx], Y_alphas[idx] ) ]

plt.contourf( X_alphas, Y_alphas, dL2, cmap='brg' )
plt.xlabel('alpha 1')  # Label for the x-axis
plt.ylabel('alpha 2')  # Label for the y-axis
plt.title('Contour Plot of d2')  # Title for the plot
plt.colorbar()  # Add a color bar to indicate the function values
plt.show()