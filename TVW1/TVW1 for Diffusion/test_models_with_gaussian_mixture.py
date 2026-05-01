# %% build_gaussian_mixture (etc...)
import numpy as np

def function_as_array(coordinates, f):
    # Create meshgrid in "matrix" indexing mode (not 'xy')
    mesh = np.meshgrid(*coordinates, indexing='ij')
    # Use vectorized function to evaluate f on each combination
    return np.vectorize(f)(*mesh)

def gaussian_density(mu, sig_inv, *x):
    mu, x = np.asarray(mu), np.asarray(x)
    d = len(mu)
    exponent = - 0.5 * (x - mu).T @ sig_inv @ (x - mu)
    const = (2 * np.pi) ** (d / 2)
    return np.exp(exponent) * np.sqrt(np.abs(np.linalg.det(sig_inv))) / const

def gaussian_density_factory(mu, sig_inv):
    mu = np.asarray(mu)
    def f(*x):
        return gaussian_density(mu, sig_inv, *x)
    return f

def rotation_matrix_2d(phi):
    return np.array( [[ np.cos(phi), -np.sin(phi) ],
                      [ np.sin(phi), np.cos(phi) ]] )

def Givens_rotation(n, i, j, phi):
    R = np.eye(n)
    c, s = np.cos(phi), np.sin(phi)
    R[[i,i,j,j],[i,j,i,j]] = c, -s, s, c
    return R

def orthonormal_basis_from_angles(n, phis):
    assert len(phis) == n*(n-1)//2
    U = np.eye(n)
    idx = 0
    for i in range(n-1):
        for j in range(i+1,n):
            U = Givens_rotation(n, i, j, phis[idx]) @ U
            idx += 1
    return U.T

def generate_field_2d(shape_x, a, b, c=0):
    """Generates a (N,N) array of angles where the flow is symmetric 
    along the lower-left to upper-right diagonal."""
    M, N = shape_x
    X, Y = np.meshgrid(np.linspace(-1, 1, M), np.linspace(-1, 1, N), indexing='ij')

    # Compute angles relative to the diagonal symmetry axis
    A = np.pi / 4 * (a*X + b*Y + c)  # Symmetric variation along the diagonal

    return A.reshape( A.shape + (1,))

def generate_gaussians_from_eigval_and_field(eigenvalues, field, shape_y):
    ndim_y = len(shape_y)
    number_of_angles = field.shape[-1]
    assert number_of_angles == ndim_y*(ndim_y-1)//2
    shape_x = field.shape[:-1]
    P = np.zeros( shape_x + shape_y )
    coordinates = [ np.linspace( -8, 8, n ) for n in shape_y ]
    Delta = np.diag(eigenvalues)
    for idx in np.ndindex(shape_x):
        U = orthonormal_basis_from_angles(ndim_y, field[idx] )
        sig = U.T @ Delta @ U
        f = gaussian_density_factory( ndim_y*[0], np.linalg.inv(sig)  )
        P[idx] = function_as_array( coordinates, f )
    return P

def build_gaussian_mixture(shape_x, shape_y):
    field_1 = generate_field_2d(shape_x, 0.8, 0.8, 2)
    field_2 = generate_field_2d(shape_x, 0, 0, -0.75)
    eigenvalues = (8, 0.6)
    gaussians_1 = generate_gaussians_from_eigval_and_field(eigenvalues, field_1, shape_y)
    gaussians_2 = generate_gaussians_from_eigval_and_field(eigenvalues, field_2, shape_y)
    gaussian_mixture = gaussians_1 + gaussians_2
    return gaussian_mixture

# %% undersampling_masks

def undersampling_mask_y(shape_y, retained_ratio, concentration_coeff):
    assert 0 < concentration_coeff < 1
    ndim_y = len(shape_y)
    size_y = np.prod(shape_y)
    N_ones_y = int(retained_ratio * size_y)
    f = gaussian_density_factory(ndim_y*[0], np.eye(ndim_y))
    domain = 4 * concentration_coeff * np.pi/2
    coordinates = [ np.linspace(-domain, domain, n) for n in shape_y ]
    probabilities_flat = function_as_array(coordinates, f).flatten()
    probabilities_flat = probabilities_flat / np.sum(probabilities_flat)
    indices_flat = np.random.choice( size_y, N_ones_y, replace=False, p=probabilities_flat )
    indices = np.unravel_index( indices_flat, shape_y )
    mask_y = np.zeros(shape_y, dtype=bool)
    mask_y[indices] = True
    return mask_y

def undersampling_mask_xy(shape_x, shape_y, retained_ratio, concentration_coeff):
    mask_xy = np.zeros( shape_x + shape_y, dtype=bool)
    mask_y = undersampling_mask_y(shape_y, retained_ratio, concentration_coeff)
    mask_xy[:] = mask_y # using NumPy's broadcasting
    return mask_xy

# %% Build data E and plot KKstar(E)

shape_x = (7, 8)
shape_y = (19, 20)

gaussian_mixture = 100 * build_gaussian_mixture(shape_x, shape_y) # !! se non uso il fattore 100 la ricostruzione con L2_TVW1_naif ha dei problemi
gaussian_mixture[ gaussian_mixture < 1e-14 ] = 0

ndim_x, ndim_y = len(shape_x), len(shape_y)
shape_xy = shape_x + shape_y
ndim_xy = len(shape_xy)

retained_ratio = 0.15
mask_fft = undersampling_mask_xy(shape_x, shape_y, retained_ratio, concentration_coeff=0.7)
mask_fft = np.fft.ifftshift(mask_fft)
mask_rfft = mask_fft[tuple(slice(None) if i != ndim_xy-1 else slice(0, shape_xy[-1]//2 + 1) 
                    for i in range(ndim_xy))]

from L2_TVW1_naif__model import KK_factory, KKstar_factory

KK, KKstar = KK_factory(mask_rfft), KKstar_factory(mask_rfft)

E_clean = KK( gaussian_mixture )

size_E = E_clean.size
E_energy = np.sum( np.abs(E_clean)**2 )
SNR = 20
standard_deviation = np.sqrt( E_energy / ( size_E * SNR ) )
noise = np.random.normal( size=size_E, scale=standard_deviation )
E = E_clean + noise

if __name__ == '__main__':
    from plottings import plot_u
    plot_u( gaussian_mixture, ndim_y, title="gaussian_mixture" )
    title = f"rough reconstruction, SNR={SNR}, retained_rato={retained_ratio}"
    plot_u( KKstar( E ).clip(min=0), ndim_y, title )

# %% Test model_naif_graph

from L2_TVW1_naif__model import model_naif_graph

if __name__ == '__main__':

    alpha1 = 0.45
    alpha2 = 1.4
    iterations = 4000
    sigma=1

    N = 4  # operators A_0, A_1, A_2, A_3

    # state graph (complete DAG)
    E_state = [(0,1), (0,2), (0,3),
            (1,2), (1,3),
            (2,3)]

    parent_node = [[] for _ in range(N)]
    for h, i in E_state:
        parent_node[i].append(h)

    # parent_node == [[], [0], [0, 1], [0, 1, 2]]

    # base graph (tree / chain)
    E_base = [(0,1), (1,2), (2,3)]   # N-1 edges

    # incidence matrix Z: for edge e=(u,v), column has -1 at u and +1 at v
    Z = np.zeros((N, N-1))
    for j, (u, v) in enumerate(E_base):
        Z[u, j] = -1
        Z[v, j] = +1

    # # Laplacian and degrees (diagonal)
    # L = Z @ Z.T

    adj = [set() for _ in range(N)]
    for h,i in E_state:
        adj[h].add(i)
        adj[i].add(h)
    d = np.array([len(a) for a in adj], dtype=float)
    """
    For this choice:
    Z =
    [[-1,  0,  0],
     [ 1, -1,  0],
     [ 0,  1, -1],
     [ 0,  0,  1]]
    
    L =
    [[ 1, -1,  0,  0],
     [-1,  2, -1,  0],
     [ 0, -1,  2, -1],
     [ 0,  0, -1,  1]]
    
    d = [3, 3, 3, 3]
    """

    graph_DR_parameters = ( Z, parent_node, d )

    Pstar = model_naif_graph(E, mask_rfft, shape_x, shape_y, alpha1, alpha2, graph_DR_parameters, sigma, iterations)
    
    title = f"model: naif_graph, retained={int(retained_ratio*100)}%, SNR={SNR},  alpha1={alpha1}, alpha2={alpha2}, iterations={iterations}"
    plot_u(Pstar, ndim_y, title=title )


# %% Test model_naif

from L2_TVW1_naif__model import model_naif

if __name__ == '__main__':

    alpha1 = 0.45
    alpha2 = 1.4
    stepsize_ratio = 0.5
    iterations = 1000

    Pstar = model_naif( E, mask_rfft, shape_x, shape_y, alpha1, alpha2, stepsize_ratio, iterations )
    
    title = f"model: naif, retained={int(retained_ratio*100)}%, SNR={SNR},  alpha1={alpha1}, alpha2={alpha2}, iterations={iterations}"
    plot_u(Pstar, ndim_y, title=title )

# %% test L2_TVW1_basic__model

from L2_TVW1_basic__model import model_basic

if __name__ == '__main__':

    alpha = 0.45
    beta = 0
    stepsize_ratio = 0.5
    iterations = 15000

    Pstar = model_basic( E, mask_rfft, shape_x, shape_y, alpha, beta, stepsize_ratio, iterations )
    
    title = f"model: basic, retained={int(retained_ratio*100)}%, SNR={SNR},  alpha={alpha}, beta={alpha}, it={iterations}"
    plot_u(Pstar, ndim_y, title=title )

