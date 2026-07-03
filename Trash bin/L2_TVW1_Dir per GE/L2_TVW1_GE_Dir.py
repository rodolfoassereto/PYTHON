import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from plottings import plot_u
import time

np.random.seed(0)


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
    f = lambda *x: gaussian_density(mu, sig_inv, *x)
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
    angles_field = np.pi / 4 * (a*X + b*Y + c)  # Symmetric variation along the diagonal

    return angles_field.reshape( angles_field.shape + (1,))

def visualize_field_2d(angles_field):
    
    M, N, _ = angles_field.shape
    X, Y = np.meshgrid(np.linspace(-1, 1, M), np.linspace(-1, 1, N), indexing='ij')
    theta = angles_field[:, :, 0]
    U = np.cos(theta)
    V = np.sin(theta)
    
    # Plot quiver
    plt.figure(figsize=(6, 5))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=5)
    plt.gca().set_aspect('equal')
    plt.show()

def generate_field_3d(shape_x, ndim_y, coeff):
    
    assert coeff.shape == (ndim_y, 4)
    
    field_euler_angles = np.zeros( shape_x + (ndim_y,) )
    meshgrid_output = np.meshgrid( *[ np.linspace(0,1,n) for n in shape_x ], indexing='ij')
    
    X = meshgrid_output[0]
    Y = meshgrid_output[1]
    
    output = np.zeros( shape_x + (ndim_y,) )
    for i in range(ndim_y):
        output[...,i] = coeff[i,0] + coeff[i,1]*X + coeff[i,2]*Y + coeff[i,3]*X*Y
    return output

def visualize_vector_field_3d(field_euler_angles, arrows_length=0.1):
    '''
    Use this function if ndim_y=3 (ndim_x can be 2 or 3 indifferently)
    
    field_euler_angles.shape comes from generate_field and has shape (shape_x, 3)
    '''
    
    from scipy.spatial.transform import Rotation as R
    import pyvista as pv
    
    assert field_euler_angles.ndim in {3,4}
    # if ndim_x=2, I reshape field_euler_angles to visualize it
    if field_euler_angles.ndim == 3:
        field_euler_angles = field_euler_angles.reshape( (1,) + field_euler_angles.shape )
    
    M, N, P, _ = field_euler_angles.shape
    # Convert Euler angles to vectors
    reshaped = field_euler_angles.reshape(-1, 3)
    ## Create rotation objects
    r = R.from_euler('ZYX', reshaped)
    ## Apply rotations to z-axis unit vector
    z_unit = np.array([0, 0, 1])
    vectors = r.apply(z_unit)
    vectors = vectors.reshape(M, N, P, 3)

    # Create the grid points
    X, Y, Z = np.meshgrid(
        np.linspace(0, 1, M),
        np.linspace(0, 1, N),
        np.linspace(0, 1, P),
        indexing='ij'
    )

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    vecs = vectors.reshape(-1, 3)

    # Create a PyVista point cloud
    pdata = pv.PolyData(points)
    pdata["vectors"] = vecs

    # Add glyphs (arrows) to show direction
    arrows = pdata.glyph(orient="vectors", scale=False, factor=arrows_length)

    plotter = pv.Plotter()
    plotter.add_mesh(arrows, color="blue")
    plotter.show()


def generate_gaussians_from_eigval_and_field(eigenvalues, field, shape_y):
    '''
    Parameters:
    eigenvalues : ndim_y eigenvalues
    field : array shaped shape_x + (1,) (result of generate_field, if I recall correctly the last coordinate stores the rotation angle (or Givens angles if 3d))
    shape_y : shape_y

    Returns: P gaussians
    '''
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

# Note: gaussian_mixture won't sum to 1, but they'll approximately sum to the same number.
def build_gaussian_mixture(shape_x, shape_y):
    if len(shape_y) == 2:
        field_1 = generate_field_2d(shape_x, 0.8, 0.8, 2)
        field_2 = generate_field_2d(shape_x, 0, 0, -0.75)
        eigenvalues = (8, 0.6)
    if len(shape_y) == 3:
        coeff_1 = 1 * np.ones((3,4))
        coeff_1[0,:] = 0
        field_1 = generate_field_3d( shape_x, 3, coeff_1 )
        coeff_2 = np.zeros((3,4))
        coeff_2[0,:] = 1
        field_2 = generate_field_3d( shape_x, 3, coeff_2 )
        eigenvalues = (8, 0.5, 0.5)
    gaussians_1 = generate_gaussians_from_eigval_and_field(eigenvalues, field_1, shape_y)
    gaussians_2 = generate_gaussians_from_eigval_and_field(eigenvalues, field_2, shape_y)
    gaussian_mixture = gaussians_1 + gaussians_2
    return gaussian_mixture


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

###############################################################################

def II(P, shape_x, shape_y):
    assert P.shape == shape_x + shape_y
    axes_to_sum = tuple( [ -(i+1) for i in range(len(shape_y)) ] )
    return np.sum( P, axis=axes_to_sum )
# note: the spectral norm of II is exactly size_y = np.prod(shape_y)

def IIstar(x, shape_y):
    ndim_y = len(shape_y)
    x1 = x.reshape( x.shape + ndim_y * (1,) )
    return np.broadcast_to(x1, x.shape + shape_y)

def JJ(P, shape_x, shape_y):
    N = np.prod(shape_y)
    return P - IIstar( II(P, shape_x, shape_y)/N, shape_y )

JJstar = JJ # JJ is self adjoint


def model_Dir(E, mask_rfft, shape_x, shape_y, alpha1, alpha2, beta, stepsize_ratio, iterations, printprogress=True):
    
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from numpy.random import rand as rd
    
    ndim_x, ndim_y = len(shape_x), len(shape_y)
    shape_P = shape_x + shape_y
    P0 = rd( *shape_P )
    shape_f = nabla_x( P0, dim=ndim_x ).shape
    f0 = rd( *shape_f )
    shape_Q = nabla_y( f0, dim=ndim_y  ).shape
    Q0 = rd( *shape_Q )
    shape_g = nabla_x(II(P0, shape_x, shape_y), dim=ndim_x).shape
    g0 = rd( *shape_g )
    shape_h = nabla_y( P0, dim=ndim_y ).shape
    h0 = rd( *shape_h )
    shape_l = E.shape
    l0 = rd( *shape_l) + 1j* rd(*shape_l)
    
    x0 = { 'P': P0, 'Q': Q0 }
    u0 = { 'f': f0, 'g': g0, 'h': h0, 'l': l0 }
    
    def KK(P):
        assert np.isrealobj(P) # !! rfftn only if P is real
        return np.fft.rfftn(P, norm='ortho')[mask_rfft]
    def KKstar(l, shape_xy):
        full_array = np.zeros_like( mask_rfft, dtype=l.dtype )
        full_array[mask_rfft] = l
        return np.fft.irfftn( full_array, s=shape_xy, axes=list(range(len(shape_xy))), norm='ortho' )

    from prox_and_proj import prox_norm21, proj_infty_ball
    
    prox_f =     lambda x, tau:   { 'P': x['P'].clip(min=0), # !! do we need .real or .clip(min=0)?
                                    'Q': prox_norm21( x['Q'], tau*alpha1, 0 ) }
    prox_gstar = lambda u, sigma: { 'f': u['f'],
                                    'g': proj_infty_ball( u['g'] / alpha2 ),
                                    'h': u['h'] / ( 1 + sigma/beta ),
                                    'l': ( u['l'] - sigma*E ) / (1+sigma) }
    L =     lambda x: { 'f': nabla_x( JJ( x['P'], shape_x, shape_y ), dim=ndim_x ) + div_y( x['Q'] ),
                        'g': nabla_x( II( x['P'], shape_x, shape_y ), dim=ndim_x ),
                        'h': nabla_y( x['P'], dim=ndim_y ),
                        'l': KK( x['P'] ) }
    Lstar = lambda u: { 'P': -JJstar( div_x( u['f'] ), shape_x, shape_y ) - IIstar( div_x( u['g'] ), shape_y) - div_y(u['h']) + KKstar( u['l'], shape_xy ) ,
                        'Q': -nabla_y( u['f'], dim=ndim_y ) }
    size_y = np.prod(shape_y)
    L_norm_sq = 4*ndim_x + np.sqrt(size_y)*4*ndim_x + 4*ndim_y + 1 + 4*ndim_y
    sigma = 1 / np.sqrt( stepsize_ratio * L_norm_sq )
    tau = stepsize_ratio * sigma * 2 # !! multiplied by 2
    
    from algorithms_general import CP
    
    x, u = CP( x0, u0, tau, sigma, prox_f, prox_gstar, L, Lstar, iterations, printprogress=printprogress)
    
    return x['P']

###############################################################################

shape_x = (7, 8)
shape_y = (19, 20)
ndim_y = len(shape_y)
shape_xy = shape_x + shape_y
ndim_xy = len(shape_xy)

retained_ratio = 0.15                                                          # UNDERSAMPLING
mask_fft = undersampling_mask_xy(shape_x, shape_y, retained_ratio, concentration_coeff=0.7)
mask_fft = np.fft.ifftshift(mask_fft)
mask_rfft = mask_fft[ tuple(slice(None) if i != ndim_xy-1 else slice(0, shape_xy[-1]//2 + 1) 
                      for i in range(ndim_xy)) ]

def KK(P):
    assert np.isrealobj(P) # !! rfftn only if P is real
    return np.fft.rfftn(P, norm='ortho')[mask_rfft]
def KKstar(l, shape_xy):
    full_array = np.zeros_like( mask_rfft, dtype=l.dtype )
    full_array[mask_rfft] = l
    return np.fft.irfftn( full_array, s=shape_xy, axes=list(range(len(shape_xy))), norm='ortho' )

gaussian_mixture = 100 * build_gaussian_mixture(shape_x, shape_y)
gaussian_mixture[ gaussian_mixture < 1e-14 ] = 0
plot_u(gaussian_mixture, ndim_y)

E_clean = KK( gaussian_mixture )

from objective_functions import build_2d_cost_matrix
M = build_2d_cost_matrix( *shape_y )
Pstar_dict = {} # Pstar's are stored here (as a function of alpha1, beta)
minima = {}     # 


size_E = E_clean.size
E_energy = np.sum( np.abs(E_clean)**2 )
SNRs = np.array( [3.5,4,5,6,7,8,9,10,12,15,20,50,100] )                                                    # NOISE PARAMETERS
for SNR in SNRs:
    print('current SNR: ', SNR)
    standard_deviation = np.sqrt( E_energy / ( size_E * SNR ) )
    noise = np.random.normal( size=size_E, scale=standard_deviation ) + 1j* np.random.normal( size=size_E, scale=standard_deviation )
    E = E_clean + noise/np.sqrt(2)
    
    plot_u( KKstar( E, shape_xy ).clip(min=0), ndim_y, title=f"rough reconstruction with SNR={SNR}" )
    
    # # 2-D experiment:
    # # retained_ratio=0.15, SNR=10, shape_x=(7,8), shape_y=(19,20)
    # # alpha1=alpha2=0.6, beta=0.01, stepsize_ratio=1
    alpha1 = 0.4
    alpha2 = alpha1
    beta = 0.01
    stepsize_ratio = 1
    
    # 3-D experiment: shape_x=(3,6,7), shape_y=(16, 17, 18), takes 15 minutes
    # alpha1 = 0.025
    # alpha2 = alpha1
    # beta = 0.0001
    # stepsize_ratio = 1
    
    '''
    iterations = 1000
    
    Pstar = model_Dir( E, mask_rfft, shape_x, shape_y, alpha1, alpha2, beta, stepsize_ratio, iterations )
    title = f"reconstruction with SNR={SNR}, retained={int(retained_ratio*100)}%, SNR={SNR},  alpha1={alpha1}, alpha2={alpha2}, beta={beta}, stepsize_ratio={stepsize_ratio}, iterations={iterations}"
    plot_u(Pstar, ndim_y, title=title )
    '''
    
    ###############################################################################
    
    # fact1 = 1.15
    # range_alpha1 = fact1 ** np.array( range(-2,2) ) * alpha1
    # fact2 = 1.2
    # range_alpha2 = fact2 ** np.array( range(-5,5) ) * alpha2
    # fact3 = 1.2
    # range_beta = fact3 ** np.array( range(-2,2) ) * beta
    
    delta1 = 0.01
    range_alpha1 = np.arange( alpha1 - 30*delta1, alpha1 + 30*delta1, delta1 )
    
    range_alpha2 = range_alpha1.copy()
    
    delta3 = 0.0005
    range_beta = np.arange( beta - 19*delta3, beta + 20*delta3, delta3 )
    
    counter = 0
    t0 = time.time()
    iterations_short=800
    dist_L2 = {}
    for alpha1, beta in it.product( range_alpha1, range_beta ):
        alpha2 = alpha1.copy()
        Pstar_dict[(SNR, alpha1, beta)] = model_Dir( E, mask_rfft, shape_x, shape_y, alpha1, alpha2, beta, stepsize_ratio, iterations_short, printprogress=False )
        dist_L2[(alpha1, beta)] = np.sum( ( Pstar_dict[(SNR, alpha1, beta)] - gaussian_mixture )**2 )
        t1 = time.time()
        if t1 - t0 >= 60:
            print('SNR=',SNR, ': ', round(counter*100 / (len(range_alpha1)*len(range_beta)), 1), "%  (", counter, "/", (len(range_alpha1)*len(range_beta)), ")")
            t0 = t1
        counter += 1
    
    argmin_key = min(dist_L2, key=dist_L2.get)
    min_val = dist_L2[argmin_key]
    minima[SNR] = (argmin_key, min_val)


alphas_star, betas_star = [], []
for SNR in minima.keys():
    alpha_tmp, beta_tmp = minima[SNR][0]
    alphas_star += [alpha_tmp]
    betas_star += [beta_tmp]

plt.plot(alphas_star, betas_star, '-o')  # '-o' joins points with lines and marks them
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.title(r'Trajectory of $(\alpha^*, \beta^*)$')
plt.grid(True)
for i, snr in enumerate(SNRs):
    plt.text(alphas_star[i], betas_star[i], f'{snr}', fontsize=9,
             ha='right', va='bottom')
plt.show()

'''
X_alphas, Y_alphas = np.meshgrid(range_alpha1, range_beta, indexing='ij')
dL2 = np.zeros(X_alphas.shape)
dW1 = np.zeros(X_alphas.shape)
dW2 = np.zeros(X_alphas.shape)
for idx in np.ndindex(*dL2.shape):
    dL2[idx] = dist_L2[ ( X_alphas[idx], Y_alphas[idx] ) ]

plt.contourf( X_alphas, Y_alphas, dL2, cmap='brg' )
plt.xlabel('alpha 1')  # Label for the x-axis
plt.ylabel('beta')  # Label for the y-axis
plt.title('Contour Plot of d2')  # Title for the plot
plt.colorbar()  # Add a color bar to indicate the function values
plt.show()

argmin_index_flat = np.argmin( dL2 ) # it always return a single value
argmin_index = np.unravel_index( argmin_index_flat, dL2.shape )
min_dL2 = dL2[ argmin_index ]
'''