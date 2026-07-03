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

def build_mass_imbalance_factor(shape_x, shape_y, scheme="jump"):
    assert len(shape_x)==2
    
    mass_imbalance_factor = np.zeros(shape_x)
    dim1, dim2 = shape_x
    for i, j in it.product( range(dim1), range(dim2) ):
        if scheme=='jump':
            if i<=j:
                mass_imbalance_factor[i,j] = 1.5
            else: mass_imbalance_factor[i,j] = 0.35
        if scheme=='continuous':
            mass_imbalance_factor[i,j] = 0.5 + (i + 2*j) / ( 2*( (dim1-1) + 2*(dim2-1) ) )    
    return mass_imbalance_factor.reshape( shape_x + tuple(np.ones(len(shape_y), dtype=int)) )

# Note: gaussian_mixture won't sum to 1, but they'll approximately sum to the same number.
def build_gaussian_mixture(shape_x, shape_y, mass_imbalance_factor=1):
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
    gaussian_mixture = gaussians_1 + mass_imbalance_factor * gaussians_2
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

def KK(P, mask): # mask is mask_rfft
    assert np.isrealobj(P) # !! rfftn only if P is real
    return np.fft.rfftn(P, norm='ortho')[mask]
def KKstar(l, shape_xy, mask): # mask is mask_rfft
    full_array = np.zeros_like( mask, dtype=l.dtype )
    full_array[mask] = l
    return np.fft.irfftn( full_array, s=shape_xy, axes=list(range(len(shape_xy))), norm='ortho' )


def model_Dir(E, mask_rfft, shape_x, shape_y, alpha1, alpha2, beta, stepsize_ratio, iterations, printprogress=True):
    assert stepsize_ratio < 1
    
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

    from prox_and_proj import prox_norm21, proj_infty_ball
    
    prox_f =     lambda x, tau:   { 'P': x['P'].clip(min=0), # !! do we need .real or .clip(min=0)?
                                    'Q': prox_norm21( x['Q'], tau*alpha1, 0 ) }
    prox_gstar = lambda u, sigma: { 'f': u['f'],
                                    'g': proj_infty_ball( u['g'] , alpha2 ),
                                    'h': u['h'] / ( 1 + sigma*beta ),
                                    'l': ( u['l'] - (sigma**2)*E ) / (1+sigma) }
    L =     lambda x: { 'f': nabla_x( JJ( x['P'], shape_x, shape_y ), dim=ndim_x ) + div_y( x['Q'] ),
                        'g': nabla_x( II( x['P'], shape_x, shape_y ), dim=ndim_x ),
                        'h': beta * nabla_y( x['P'], dim=ndim_y ),
                        'l': KK( x['P'], mask_rfft ) }
    Lstar = lambda u: { 'P': -JJstar( div_x( u['f'] ), shape_x, shape_y ) - IIstar( div_x( u['g'] ), shape_y) - beta*div_y(u['h']) + KKstar( u['l'], shape_xy, mask_rfft ) ,
                        'Q': -nabla_y( u['f'], dim=ndim_y ) }
    size_y = np.prod(shape_y)
    L_norm_sq = 4*ndim_x + size_y*4*ndim_x + 4*ndim_y + 1 + 4*ndim_y*beta
    sigma = 1 / np.sqrt( stepsize_ratio * L_norm_sq )
    tau = stepsize_ratio * sigma * 2 # !! multiplied by 2, incorrect but possibly faster
    
    from algorithms_general import CP
    
    x, u = CP( x0, u0, tau, sigma, prox_f, prox_gstar, L, Lstar, iterations, printprogress=printprogress)
    
    return x['P']

# %% ##############################################################################

shape_x = (7, 8)
shape_y = (19, 20)

ndim_y = len(shape_y)
shape_xy = shape_x + shape_y
ndim_xy = len(shape_xy)

retained_ratio = 0.3                                                           # UNDERSAMPLING
mask_fft = undersampling_mask_xy(shape_x, shape_y, retained_ratio, concentration_coeff=0.7)
mask_fft = np.fft.ifftshift(mask_fft)
mask_rfft = mask_fft[ tuple(slice(None) if i != ndim_xy-1 else slice(0, shape_xy[-1]//2 + 1) 
                      for i in range(ndim_xy)) ]

mass_imbalance_factor = build_mass_imbalance_factor(shape_x, shape_y)

gaussian_mixture = 100 * build_gaussian_mixture(shape_x, shape_y, mass_imbalance_factor=mass_imbalance_factor)
gaussian_mixture[ gaussian_mixture < 1e-14 ] = 0
plot_u(gaussian_mixture, ndim_y)

E_clean = KK( gaussian_mixture, mask_rfft )
size_E = E_clean.size
E_energy = np.sum( np.abs(E_clean)**2 )


# %%

SNRs = np.array( [10, 11, 13, 16, 20, 30, 40, 60, 100] )                          # NOISE PARAMETERS

def generate_new_grid( center, radius_L, radius_R, num_gridpoints ):
    return np.linspace( max(center-radius_L, 1e-8), center+radius_R, num_gridpoints )


stepsize_ratio = 1
iterations_short = 600

number_of_searches = 7
num_gridpoints = 5

minima = {}
Pstars_dict = {}
    
for SNR in SNRs:
    print('current SNR: ', SNR)
    
    standard_deviation = np.sqrt( E_energy / ( size_E * SNR ) )
    noise = np.random.normal( size=size_E, scale=standard_deviation ) + 1j* np.random.normal( size=size_E, scale=standard_deviation )
    E = E_clean + noise/np.sqrt(2)
    
    counter = number_of_searches
    
    center_alpha1s = 1
    center_alpha2s = 1
    radius_L_alpha1s, radius_R_alpha1s = 1, 1
    radius_L_alpha2s, radius_R_alpha2s = 1, 1
    
    grid_alpha1s = generate_new_grid( center_alpha1s, radius_L_alpha1s, radius_R_alpha1s, num_gridpoints )
    grid_alpha2s = generate_new_grid( center_alpha2s, radius_L_alpha2s, radius_R_alpha2s, num_gridpoints )
    
    while counter > 0:
        print('Left cycles: ', counter )
        counter -= 1
        
        dist_L2_dict_temp = {}
        Pstars_dict_temp = {}
        
        # plot_u( KKstar( E, shape_xy, mask_rfft ).clip(min=0), ndim_y, title=f"rough reconstruction with SNR={SNR}" )
        
        for alpha1, alpha2 in it.product(grid_alpha1s, grid_alpha2s):
            
            beta = 0 # !! togliere per beta search
            
            Pstar = model_Dir( E, mask_rfft, shape_x, shape_y, alpha1, alpha2, beta, stepsize_ratio, iterations_short, printprogress=False )
            dist_L2 = np.sum( ( Pstar - gaussian_mixture )**2 )
            
            dist_L2_dict_temp[ (alpha1, alpha2) ] = dist_L2
            Pstars_dict_temp[ (alpha1, alpha2) ] = Pstar
        
        argmin_key = min(dist_L2_dict_temp, key=dist_L2_dict_temp.get)
        min_val = dist_L2_dict_temp[argmin_key]
        
        center_alpha1s_old = center_alpha1s
        center_alpha2s_old = center_alpha2s
        
        center_alpha1s, center_alpha2s = argmin_key
        
        if center_alpha1s_old + radius_R_alpha1s <= center_alpha1s: div_factor1 = 1.25
        else: div_factor1 = 2.25
        if center_alpha2s_old + radius_R_alpha2s <= center_alpha2s: div_factor2 = 1.25
        else: div_factor2 = 2.25
        
        radius_L_alpha1s, radius_R_alpha1s = radius_L_alpha1s/2.25, radius_R_alpha1s/div_factor1
        radius_L_alpha2s, radius_R_alpha2s = radius_L_alpha2s/2.25, radius_R_alpha2s/div_factor2
        
        grid_alpha1s = generate_new_grid( center_alpha1s, radius_L_alpha1s, radius_R_alpha1s, num_gridpoints )
        grid_alpha2s = generate_new_grid( center_alpha2s, radius_L_alpha2s, radius_R_alpha2s, num_gridpoints )
    
    minima[SNR] = (argmin_key, min_val)
    Pstars_dict[SNR] = Pstars_dict_temp[ argmin_key ]


# %%

SNRs = []
x_values = []
y_values = []
min_vals = []

for SNR, (argmin_key, min_val) in minima.items():
    alpha1, alpha2 = argmin_key
    SNRs.append(SNR)
    x_values.append(alpha1)
    y_values.append(alpha2)
    min_vals.append(min_val)

SNRs     = np.array(SNRs)
x_values  = np.array(x_values)
y_values    = np.array(y_values)
min_vals = np.array(min_vals)

# --- Plot ---
fig, ax = plt.subplots()

sc = ax.scatter(x_values, y_values, c=min_vals, s=60, edgecolors='k')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("min_value")

ax.set_xlabel("alpha1")
ax.set_ylabel("alpha2")
ax.set_title("Best (alpha1, alpha2) per SNR, colored by min_value")

# --- Add SNR labels next to each point ---
for x, y, SNR in zip(x_values, y_values, SNRs):
    ax.text(x+0.0025, y+0.00025, str(SNR), fontsize=8)  # small offset so text doesn't sit on top

plt.tight_layout()
plt.show()

