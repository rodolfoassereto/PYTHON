import numpy as np
from numpy.random import rand as rd

# %% This cell tests the structure of the system (id - ∆ )x = b  (with a sparse laplacian)

import scipy.sparse as sp
from differential_operators_sparse import create_partials_sparse, create_nabla_sparse, create_laplacian_sparse

shape_x, shape_y = (3,4), (3,5)
shape_xy = shape_x + shape_y
ndim_x, ndim_y = len(shape_x), len(shape_y)
n_x, n_y = np.prod(shape_x), np.prod(shape_y)
n = n_x * n_y

id_x, id_y = sp.eye(n_x), sp.eye(n_y)
id_xy = sp.eye(n) 

d0, d1 = create_partials_sparse( shape_xy, (0,1), boundary='no')
d0_small, d1_small = create_partials_sparse( shape_x, (0,1), boundary='no')

N_x, L_x = create_nabla_sparse( d0, d1 ), create_laplacian_sparse( d0, d1 )
N_small, L_small = create_nabla_sparse( d0_small, d1_small ), create_laplacian_sparse( d0_small, d1_small )


I = np.ones(n_y).reshape(1,n_y)
II = sp.kron( id_x, I )

P = I.T @ I / n_y
PP = sp.kron( id_x, P )

J = sp.eye(n_y) - I.T @ I / n_y
JJ = sp.kron( id_x, J )

A = JJ.T @ L_x @ JJ

c = rd(*shape_xy).ravel()
c_P, c_J = PP @ c, JJ @ c

# z_J, _ = sp.linalg.cg( id_xy - L_x , c_J)
# z = c_P + z_J
# print( '1:', np.allclose( (id_xy - A) @ z , c ) )

# %% This cell tests the "resolvent_with_laplacian" function

from solve_linear_systems import laplacian_eigenvalues, resolvent_with_laplacian
from differential_operators import laplacian_x

shape_x, shape_y = (3,4), (3,5)
shape_xy = shape_x + shape_y

b = np.random.rand(*shape_xy)

eigenvalues = laplacian_eigenvalues(shape_xy, axes=(0,1))

u = resolvent_with_laplacian(b, 1 - 0.3*eigenvalues, (0,1), sigma=None )

print( 'Check resolvent_with_laplacian: ', np.allclose( u - 0.3*laplacian_x(u, 2), b ) )

# %% This cell tests the "resolvent_with_II" function

from L2_TVW1_naif__model import II, IIstar, JJ
from solve_linear_systems import resolvent_with_II

shape_x, shape_y = (3,4), (3,5)
shape_xy = shape_x + shape_y
size_y = np.prod(shape_y)

b = rd(*shape_xy)

sig = 0.3

eigenvalues = laplacian_eigenvalues(shape_x, (0,1))
pointwise_division = 1 - sig*size_y*eigenvalues
z = resolvent_with_II(b, II, JJ, shape_x, shape_y, pointwise_division, sigma=None)

IIz = II(z, shape_x, shape_y)
print( 'Check resolvent_with_II: ', np.allclose( z - sig*IIstar( laplacian_x(IIz, 2), shape_y), b ) )

# %% This cell tests the "resolvent_with_JJ" function

from L2_TVW1_naif__model import PP, JJ, JJstar
from solve_linear_systems import resolvent_with_JJ
from differential_operators import laplacian_x

shape_x, shape_y = (3,4), (3,5)
shape_xy = shape_x + shape_y

b = rd(*shape_xy)

sig = 0.3
eigenvalues = laplacian_eigenvalues(shape_xy, (0,1))
pointwise_division = 1 - sig  * eigenvalues
z = resolvent_with_JJ(b, PP, JJ, shape_x, shape_y, (0,1), pointwise_division, sigma=None)
JJz = JJ(z, shape_x, shape_y)

print( 'Check resolvent_with_JJ: ', np.allclose( z - sig*JJstar( laplacian_x(JJz, 2), shape_x, shape_y), b ) )