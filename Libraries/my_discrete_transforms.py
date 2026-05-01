import numpy as np
import sys
# Add Libraries folder to sys.path
libraries_path = r"C:\Users\rodol\My Drive\PYHTON\Libraries"
if libraries_path not in sys.path:
    sys.path.insert(0, libraries_path)

def discrete_cosine_1(n,k):
    return np.array( [ np.cos(np.pi*k*j/(n-1)) for j in range(n) ] )

def discrete_cosine_2(n,k):
    return np.array( [ np.cos(np.pi*k*(j+0.5)/n) for j in range(n) ] )


def DCT_1(n):
    '''
    Note: the matrix DCT is *not* orthogonal in the standard sense but only w.r.t. the inner product where entries 0 and n-1 are weighted 0.5 (instead of 1)
    '''
    DCT = np.zeros((n,n))
    for k in range(n):
        DCT[k,:] = discrete_cosine_1(n,k)
    return DCT

def DCT_2(n, orthonormal=True):
    DCT = np.zeros((n,n))
    for k in range(n):
        if orthonormal:
            temp = discrete_cosine_2(n,k)
            temp = temp / np.linalg.norm(temp)
        DCT[k,:] = temp
    return DCT

'''Note: DCT_3 is the inverse of DCT_2 (if not orthonormal a factor must be introduced)'''

# %% Check if my 1-d DCT diagonalizes my 1-d laplacian (with no boundary)

from differential_operators_sparse import create_partials_sparse, create_laplacian_sparse, create_partialplus_sparse_1d_noboundary
from numpy.random import rand as rd
import scipy.fft as spfft

# # DCT_2 diagonalizes the Laplacian
# dct2 = DCT_2(5)
# d = create_partialplus_sparse_1d_noboundary(5)
# L = -d.T @ d
# print( np.round(dct2 @ L @ dct2.T, 2) )


# %% Check:
# 1) that laplacian_x and create_laplacian_sparse implement the same laplacian (with no boundary);
# 2) that scipy's dct diagonalizes my laplacian


shape_xy = (3,4,3,5)
tmp = rd(*shape_xy)
tmp_dct = spfft.dctn( tmp, axes=(0,1), norm='ortho' ) # norm='ortho' was necessary for scipy.fftpack instead of scipy.fft, here it still works without

from solve_linear_systems import laplacian_eigenvalues
from differential_operators import laplacian_x

eigenvalues = laplacian_eigenvalues( shape_xy, axes=(0,1) )

tmp_L1 = laplacian_x(tmp, 2) # un altro modo di definire lo stesso array

partials = create_partials_sparse( shape_xy, (0,1) )
L = create_laplacian_sparse( *partials )

tmp_L = (L @ tmp.ravel()).reshape(shape_xy)

print( 'laplacian_x == create_laplacian_sparse: ', np.allclose( tmp_L1, tmp_L ) )
print( 'scipy\'s DCT diagonalizes my laplacian: ', np.allclose( spfft.idctn( eigenvalues * tmp_dct, axes=(0,1), norm='ortho' ), tmp_L ) ) # norm='ortho' was necessary for scipy.fftpack instead of scipy.fft, here it still works without

