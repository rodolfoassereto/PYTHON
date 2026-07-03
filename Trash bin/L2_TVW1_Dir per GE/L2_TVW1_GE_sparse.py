import sys, os
root_folder = "C:/Users/rodol/My Drive/PYHTON"
for root, dirs, files in os.walk(root_folder):
    sys.path.append(root)

import numpy as np
import scipy.sparse as sp
from differential_operators_sparse import create_partialplus_sparse, create_nabla_sparse, BlockLinearOperator

# alpha1, alpha2, beta, E, dim_x, dim_y
# shape_P = P0.shape

shape_P = (7,7,20,20)
dim_x, dim_y = 2,2

shape_f = (dim_x,) + shape_P
shape_Q = (dim_y,) + shape_f
shape_x = shape_P[:dim_x]

size_P = np.prod(shape_P)
size_Q = np.prod(shape_Q)
size_Px = np.prod(shape_P[:dim_x])
size_Py = np.prod(shape_P[dim_x:])

size_limit_P = 1e5
if size_P > size_limit_P:
    raise ValueError("P is too big. Stopping script.")

# create the spparse differential operators
partials_y = [ ]
partials_omega = [ ]
partials_nabla = [ ]
for i in range(dim_x):
    partials_omega += [ create_partialplus_sparse( shape_P, i) ]
    partials_nabla += [ create_partialplus_sparse( shape_x, i) ]
for i in range(dim_y):
    partials_y += [ create_partialplus_sparse( shape_f, -i ) ]

nabla_omega = create_nabla_sparse( *partials_omega )
nabla_y = create_nabla_sparse( *partials_y )
nabla = create_nabla_sparse( *partials_nabla )

# Create the operators II and JJ
I = sp.csr_matrix(np.ones((1, size_Py)))
id_x = sp.eye( size_Px, format='csr')
id_y = sp.eye( size_Py, format='csr' )
J = id_y - I.T @ I / size_Py
II = sp.kron( id_x, I, format='csr' )
JJ = sp.kron( id_x, J, format='csr' )

zero_block = sp.csr_matrix( ( nabla.shape[0] , size_Q ) )
blocks = [ [ nabla_omega @ JJ , nabla_y.T ],
           [ nabla @ II , zero_block ] ]
matrix_blocks = sp.bmat( blocks, format='csr' )

L = BlockLinearOperator(blocks)

def KK(P_flat, mask, shape):
    return mask * np.fft.rfftn(P_flat.reshape(shape), norm='ortho')

def KKstar(E):
    np.fft.irfftn( E )

from library_zero import turn_linear_into_sparse_matrix
from build_toymodels import undersampling_mask
#(shape, scheme_k='cartesian', scheme_q='gaussian', ratio_q=0.75, cov=None, **kwargs)

mask = undersampling_mask( shape_P )
# mask_sparse = sp.csr_matrix(  )

FF_sparse = turn_linear_into_sparse_matrix( lambda x: np.fft.rfftn(x), shape_P )
# KK_sparse = mask_sparse * FF_sparse

from prox_and_proj import prox_norm21
# Note: every time one takes the prox_norm21, one needs to reshape Q as
# Q1 = Q.reshape(dim_y,-1)

print('print')

s = sp.linalg.svds(id_x, k=1, which='LM', return_singular_vectors=False)[0] # spectral norm of sparse matrix

def prox_L2_term(P):
    
    sp.linalg.spsolve( KKstar(KK(P)) )
    
    return

