import sys, os
root_folder = "C:/Users/rodol/My Drive/PYHTON"
for root, dirs, files in os.walk(root_folder):
    sys.path.append(root)

import numpy as np
import matplotlib.pyplot as plt
from plottings import plot_u

img_path = 'C:/Users/rodol/My Drive/PYHTON/PYTHON_sec_3/'

''' A slice of GE dataset has size (96,96,11,11,11) which is a size of  ~12e6
We can replicate a similar order of magnitude in 4D with (100,100,33,33)'''

######## GAUSSIANS ########

def my_meshgrid(*coordinates, normalize=True, indexing='xy'):
    ''''
    Input: 1-D arrays coord_0,...,coord_d, of lengths l1,...,ld (or possibly just a shape passed as a sequence of ints)
    output: (l1,...,ld,d) array grid such that grid[i1,...,id,:] = (coord_0[i1],...,coord_d[id])
    '''
    if len(coordinates) == 1: coordinates = coordinates[0] # In this case I suppose the user is passing the desired shape as a tuple
    if all(isinstance(x, int) for x in coordinates): # in this case I suppose the user is passing a shape dim by dim
        if normalize == True: coordinates = [np.linspace(0, 1, dim) for dim in coordinates]
        else: coordinates = [np.linspace(0, dim-1, dim) for dim in coordinates]
    return np.array( np.meshgrid(*coordinates, indexing=indexing) ).T

def quadratic_function(A, q): # A is meant to be the result of my_meshgrid = moveaxis(meshgrid(coord_1,...,coord_k),0,-1)
    '''
    Computes the einsum operation np.einsum('i1...ikd,dn,i1...ikn', A, q, A) for an array A with shape (N1,...,Nk,D) and q with shape (D,D).
    Parameters: A (np.ndarray): Input array with shape (N1,...,Nk,D)
                q (np.ndarray): Matrix with shape (D,D)
    Returns:    np.ndarray: Result of the einsum operation with shape (N1,...,Nk)
    '''
    import string
    k = A.ndim - 1 # Number of free indices (excluding the last dimension D)
    if k!=len(q): raise ValueError('A s supposed to represent coordinates with same size as the side of q')
    # Generate index labels (single characters) for free indices
    indices = string.ascii_letters[:-2]  # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX'
    if k > len(indices): raise ValueError(f"Too many dimensions in A (maximum supported is {len(indices)}).")
    free_indices = indices[:k]  # e.g., 'i', 'ij', 'ijk', etc.
    # Build the einsum subscript strings
    subscript_A1 = free_indices + 'Y'   # e.g., 'iY', 'ijY', 'ijkY'
    subscript_q = 'YZ'                  # Fixed for q
    subscript_A2 = free_indices + 'Z'   # e.g., 'iZ', 'ijZ', 'ijkZ'
    # Combine into the full einsum subscript
    einsum_subscript = f'{subscript_A1},{subscript_q},{subscript_A2}->{free_indices}'
    return np.einsum(einsum_subscript, A, q, A)

def rotate_2d_array(A, theta):
    # Create the rotation matrix R(theta), useful for covariance matrix in normal distributions
    R = np.array([[np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]])
    return R @ A @ R.T # Rotate the operator by conjugating with R

def gaussian_function_as_matrix(*coordinates, mean=None, cov=None, output_dxyz=False):
    ''' Builds a matrix M such that M[i,j] = f(x_coord[i],y_coord[j]), f being the normal density with parameters mean and cov '''
    ''' Note: the plt.imshow of the output appears transposed (because coord_0 is vertical and coord_1 horizontal) '''
    d = len(coordinates)
    if mean is None: mean = np.zeros(d)
    if cov is None: cov = np.eye(d)
    if type(cov) in {float, int}: cov = cov*np.eye(d)
    coordinates_2 = [ coord - mean[k] for k, coord in enumerate(coordinates) ]
    grid = my_meshgrid(*coordinates_2)
    covinv = np.linalg.inv(cov)
    dxyz = [] # will contain the steps of the grid, supposig that the grids are evenly spaced
    density = np.sqrt( (2*np.pi)**-d * np.linalg.det(covinv) ) * np.exp( -0.5 * quadratic_function(grid, covinv) )
    if output_dxyz:
        for coord in coordinates:
            dxyz += [ (coord[-1] - coord[0]) / (len(coord)-1) ]
        dxyz = np.prod(dxyz) # measure of infinitesimal volume
        return density, dxyz
    else: return density 


####### UNDERSAMPLING MASKS ########

from skimage.draw import line

def radial_mask(shape, scheme="nomask", **kwargs): # only works for 2-D shapes
    """
    Create a k-space mask for MRI imaging based on the specified scheme.

    Parameters:
    - shape (tuple): Shape of the mask as (height, width).
    - scheme (str): Masking scheme - "nomask", "radial", or "cartesian".
    - **kwargs: Additional parameters depending on the scheme.

        For "radial":
            - num_radials (int): Number of radial lines.
            - full_radius (bool): If True, lines extend to the full edge. Default is True.

        For "cartesian":
            - gap (int): Number of pixels between parallel lines.
            - orientation (str): "horizontal" or "vertical". Default is "horizontal".

    Returns:
    - mask (numpy.ndarray): 2D binary mask with True for sampled points and False otherwise.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)

    if scheme == "nomask":
        # Return an empty mask
        return mask

    elif scheme in {"radial"}:
        height, width = height+2, width+2 # I added this because the radii didn't reach the border. Then, I truncate the output
        mask = np.zeros((height, width), dtype=bool)
        # Retrieve radial parameters with defaults
        num_radials = kwargs.get('num_radials', 30)
        full_radius = kwargs.get('full_radius', True)

        center_y, center_x = height // 2, width // 2
        radius = min(center_y, center_x)
        if not full_radius:
            radius = radius // 2

        for i in range(num_radials):
            angle = (2 * np.pi * i) / num_radials
            x_end = center_x + int(radius * np.cos(angle))
            y_end = center_y + int(radius * np.sin(angle))

            rr, cc = line(center_y, center_x, y_end, x_end)

            # Clip coordinates to mask dimensions
            rr = np.clip(rr, 0, height - 1)
            cc = np.clip(cc, 0, width - 1)

            mask[rr, cc] = True

        return mask[1:-1,1:-1]

    elif scheme in {"cartesian"}:
        # Retrieve cartesian parameters with defaults
        gap = kwargs.get('gap', 2)
        orientation = kwargs.get('orientation', 'horizontal').lower()

        if orientation not in ['horizontal', 'vertical']:
            raise ValueError("orientation must be 'horizontal' or 'vertical'.")

        if orientation == 'horizontal':
            for y in range(0, height, gap):
                mask[y, :] = True
        else:  # vertical
            for x in range(0, width, gap):
                mask[:, x] = True

        return mask

    else:
        raise ValueError(f"Unknown scheme '{scheme}'. Supported schemes are 'nomask', 'radial', and 'cartesian'.")

def symmetrize_2d_mask(msk): # !! ONLY WORKS FOR 2-D
    ''' Mette nella metà alta della matrice il simmetrico della metà bassa. Nella riga centrale, mantiene la parte destra '''
    index_bool = msk.copy()
    v = np.zeros(np.size(index_bool), dtype=bool)
    v[0:int(len(v)/2)] = True
    mskmsk = v.reshape(index_bool.shape)
    index_bool[mskmsk] = np.flip(index_bool)[mskmsk]
    return index_bool

def avoid_angles_mask(shape):
    s = np.array(shape)
    # Compute the exact geometric center (as float coordinates),
    center = (s - 1) / 2.0
    # Compute the minimal distance from the center to any "flat" edge (e.g: shape = (10,10), center = (4.5, 4.5), this is 4.5)
    edge_dists = np.minimum(center, (s - 1) - center)
    min_edge_dist = edge_dists.min()
    # Slightly inflate that distance so that the integer boundary points fall inside the mask, while corners (which are farther) remain out.
    radius = min_edge_dist + 0.5
    # Create coordinate grids and compute squared distance from center
    coords = np.ogrid[[slice(0, x) for x in s]]
    dist_sq = sum((g - c) ** 2 for g, c in zip(coords, center))
    # Build mask: points whose distance is within 'radius' => True
    mask = dist_sq <= radius ** 2
    return mask

def undersampling_mask_small(shape, ratio=0.5, scheme='gaussian', cov=None, sym=False, noangles=True, **kwargs):
    ''' Returns a "shape" mask whith exactly N 'True' entries.The Trues are sampled as a uniform or as gaussian (...) depending on the var scheme'''
    ''' "True" entries are the ones masked in numpy's masked module' '''
    ''' **kwargs are additional arguments in case the scheme is 'radial' or 'cartesian' (see the function radial_mask for details) '''
    ''' "ratio" sets the portion of True entries '''
    dim = len(shape)
    index_bool = np.zeros(shape, dtype=bool)
    N = int(np.prod(shape) * ratio) # number of True entries
    if scheme == 'nomask': index_bool = np.ones(shape, dtype=bool)
    # if scheme == 'ball': 
    if scheme == 'gaussian':
        if cov is None: cov = np.eye(dim)
        coordinates = [np.linspace(-3,3,n) for n in shape ]
        P = gaussian_function_as_matrix(*coordinates, cov=cov)
        if noangles: P[np.invert(avoid_angles_mask(shape))] = 0 # sets to 0 the probability in the corners
        P = np.ravel(P)/np.sum(P)
        index_temp = np.unravel_index( np.random.choice(np.prod(shape), N, replace=False, p=P), np.shape(index_bool) ) # chooses N indices among m*n with a 1-D probability with law p; unravel_index converts the indices to be read on the unravelled array
        index_bool[index_temp] = True
        #index_bool = np.fft.ifftshift(index_bool) (*) should I restore this? (!!)
    if scheme == 'random':
        index_temp = np.unravel_index( np.random.choice(np.prod(shape), N, replace=False), np.shape(index_bool) )
        index_bool[index_temp] = True
    if scheme in {'radial', 'cartesian'}:
        return radial_mask(shape, scheme=scheme)
    if sym == True: # !! Actually we should handle the fftshifted case? (!!) only works for 2d
        index_bool = symmetrize_2d_mask(index_bool)
    return index_bool

def undersampling_mask(shape, scheme_k='cartesian', scheme_q='gaussian', ratio_q=0.75, cov=None, **kwargs):
    shape_k, shape_q = shape[:2], shape[2:]
    mask_k = undersampling_mask_small(shape_k, scheme=scheme_k, **kwargs)
    mask_q = np.fft.fftshift(undersampling_mask_small(shape_q, scheme=scheme_q, ratio=ratio_q, cov=cov))
    import matplotlib.pyplot as plt
    plt.imshow(mask_k, cmap='gray')
    plt.show()
    # plt.imshow(mask_q, cmap='gray')
    # plt.show()
    print(np.sum(mask_q))
    print(np.sum(mask_k))
    if len(shape_q) == 2: mask_q = mask_q.reshape(shape_q + (1,))
    shape_q = mask_q.shape
    n0, n1, n2 = shape_q
    index_bool = np.zeros(shape_k + shape_q, dtype=bool)
    for j0 in range(n0):
        for j1 in range(n1):
            for j2 in range(n2):
                if mask_q[j0,j1,j2]: index_bool[:,:,j0,j1,j2] = mask_k
    return ~index_bool

#################################
########### TOYMODELS ###########
#################################

from scipy.ndimage import map_coordinates

def rotate_image(x, theta):
    """
    Rotates a 2D NumPy array by an angle theta (in radians).

    Parameters:
    x (numpy.ndarray): Input 2D array to rotate.
    theta (float): Rotation angle in radians.

    Returns:
    numpy.ndarray: Rotated 2D array.
    """
    H, W = x.shape
    cy, cx = H / 2.0, W / 2.0

    # Create coordinate grid
    y_indices, x_indices = np.indices((H, W))
    x_indices = x_indices - cx
    y_indices = y_indices - cy

    # Rotation transformation
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rot = cos_theta * x_indices + sin_theta * y_indices
    y_rot = -sin_theta * x_indices + cos_theta * y_indices

    # Shift back to original coordinate system
    x_rot += cx
    y_rot += cy

    # Interpolate using map_coordinates
    rotated = map_coordinates(
        x, [y_rot.ravel(), x_rot.ravel()], order=1, mode='constant', cval=0.0
    ).reshape((H, W))

    return rotated

########### -+|o. DISPLACEMENTS ###########

datashape = (6,6,15,15)
R1, R2, Q1, Q2 = datashape # Q1, Q2 used in displacement_factory

def displacement_factory( displ_shape, shape = (Q1,Q2), fading=True, normalize=True ):
    q1, q2 = shape
    out = np.full(shape, False)
    if displ_shape == '-':
        for i in range(q1):
            if i>=2*q1/5 and i+1<=3*q1/5:  out[i,:] = True
    if displ_shape == '|':
        for j in range(q2):
            if j>=2*q2/5 and j+1<=3*q2/5:  out[:,j] = True
    if displ_shape == '--':
        for i in range(q1):
            if i>=q1/3 and i+1<=2*q1/3:  out[i,:] = True
    if displ_shape == '||':
        for j in range(q2):
            if j>=q2/3 and j+1<=2*q2/3:  out[:,j] = True
    if displ_shape == '+': out = displacement_factory('|', shape, fading=False, normalize=False) + displacement_factory('-', shape, fading=False, normalize=False)
    if displ_shape == '++': out = displacement_factory('||', shape, fading=False, normalize=False) + displacement_factory('--', shape, fading=False, normalize=False)
    a, b = q1/2, q2/2
    if displ_shape == 'o':
        for i in range(q1):
            for j in range(q2):
                if np.linalg.norm([(i-a)/a,(j-b)/b]) <= 1 and np.linalg.norm([(i-a)/a,(j-b)/b]) >= 4/5:
                    out[i,j] = True
    if displ_shape == 'oo':
        for i in range(q1):
            for j in range(q2):
                if np.linalg.norm([(i-a)/a,(j-b)/b]) <= 1 and np.linalg.norm([(i-a)/a,(j-b)/b]) >= 3/5:
                    out[i,j] = True
    if displ_shape == '.':
        for i in range(q1):
            for j in range(q2):
                if np.linalg.norm([(i-a)/a,(j-b)/b]) <= 1/2:
                    out[i,j] = True
    if displ_shape == '..':
        for i in range(q1):
            for j in range(q2):
                if np.linalg.norm([(i-a)/a,(j-b)/b]) <= 3/4:
                    out[i,j] = True
    if fading == True:
        coord_0, coord_1 = np.linspace(-1,1,q1), np.linspace(-1,1,q2)
        P = gaussian_function_as_matrix(coord_0, coord_1, cov=2*np.eye(2))
        P = P/np.max(P)
        out = out*P
    if normalize == True:
        return out/np.sum(out)
    return out

def toydata_displacements_fromfactory():
    displ_hor = displacement_factory('-',shape=(Q1,Q2))
    displ_ver = displacement_factory('|',shape=(Q1,Q2))
    displ_plus = displacement_factory('+',shape=(Q1,Q2))
    displ_circ = displacement_factory('o',shape=(Q1,Q2))
    displ_dot = displacement_factory('.',shape=(Q1,Q2))
    displ_bigdot = displacement_factory('..',shape=(Q1,Q2))
    
    toydata = np.zeros(datashape)
    toydata[:int(R1/2),:int(R2/2)] = displ_hor
    toydata[int(R1/2):,:int(R2/2)] = displ_plus
    toydata[int(R1/2):,int(R2/2):] = displ_circ + displ_plus + displ_dot
    toydata[:int(R1/2),int(R2/2):] = displ_bigdot + displ_hor
    
    return toydata

########### "GAMMA-SHAPED" DISPLACEMENTS ###########

M_gammas=5
N_gammas=30
def build_gammashape(N=N_gammas):
    gammashape = np.zeros((N, N))
    gammashape[int(N/10):int(7*N/10), int(4.5*N/10):int(5.5*N/10)] = 1  # Shaft of the arrow
    gammashape[int(N/10):int(2*N/10), int(4.5*N/10):int(7.5*N/10)] = 1  # Head of the arrow
    gammashape[int(3*N/10):int(4*N/10), int(3.5*N/10):int(4.5*N/10)] = 1  # Head of the arrow
    return gammashape

from scipy.ndimage import shift

def build_u_gammas(M=M_gammas, N=N_gammas, angle=np.pi/6):
    u_gammas = np.zeros((M,M,N,N))
    gammashape = build_gammashape(N)
    for i in range(M):
        for j in range(M):
            x = rotate_image(gammashape, i*angle) # rotate
            u_gammas[i,j] = shift( x, shift=(0, 7*j-int(N/2.5)) ) # shift
    return u_gammas.clip(min=0)

u_gammas = build_u_gammas()
mask_inpainting = np.zeros(u_gammas.shape, dtype=bool)
mask_inpainting[1,1,:,:] = True
u_gammas_masked_inpainting = np.ma.masked_array(u_gammas, mask=mask_inpainting)

mask = np.zeros(u_gammas.shape, dtype=bool)
for i in range(M_gammas):
    for j in range(M_gammas):
        mask[i,j] = undersampling_mask_small((N_gammas,N_gammas), ratio=0.1)
mask = np.invert(np.fft.fftshift(mask))

########### STARS AND SPIRALS FOR MEDIAN EXPERIMENT ########### (see Enis & Carlier)

import matplotlib.path as mpath

def generate_star_array(N, n_points=5, outer_radius=None, inner_radius=None):
    """
    Generates a 2-D NumPy array with ones forming a star shape of adjustable size.

    Parameters:
    - N: Size of the array (NxN).
    - n_points: Number of points in the star (default is 5 for a five-pointed star).
    - outer_radius: Outer radius of the star (default is N//2 if not specified).
    - inner_radius: Inner radius of the star (default is outer_radius / 2 if not specified).

    Returns:
    - grid: 2-D NumPy array with the star shape.
    """
    grid = np.zeros((N, N), dtype=int)
    center = N / 2  # Use float for precision

    # Set default radii if not provided
    if outer_radius is None:
        outer_radius = N / 2 - 1  # Subtract 1 to prevent going out of bounds
    if inner_radius is None:
        inner_radius = outer_radius / 2

    # Generate star polygon vertices
    angles = np.linspace(0, 2 * np.pi, num=n_points * 2, endpoint=False) - np.pi / 2
    radii = np.empty(n_points * 2)
    radii[::2] = outer_radius
    radii[1::2] = inner_radius
    x = radii * np.cos(angles) + center
    y = radii * np.sin(angles) + center

    # Create a Path object for the star polygon
    vertices = np.column_stack([x, y])
    # Append the first vertex again to close the polygon
    vertices = np.vstack([vertices, vertices[0]])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(vertices) - 2) + [mpath.Path.CLOSEPOLY]

    path = mpath.Path(vertices, codes)

    # Create a grid of points
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    points = np.vstack((X.flatten(), Y.flatten())).T

    # Determine which points are inside the star polygon
    mask = path.contains_points(points)
    mask = mask.reshape((N, N))

    # Set the grid values
    grid[mask] = 1

    return grid

def generate_spiral_array(N, thickness=2, num_turns=3, max_radius=None):
    """
    Generates a 2-D NumPy array with ones forming a thick Archimedean spiral shape of adjustable size.

    Parameters:
    - N: Size of the array (NxN).
    - thickness: Thickness of the spiral lines in pixels.
    - num_turns: Number of turns in the spiral.
    - max_radius: The maximum radius of the spiral (default is N/2 - 1).

    Returns:
    - grid: 2-D NumPy array with the spiral shape.
    """
    # Initialize the grid
    grid = np.zeros((N, N), dtype=int)
    center = N / 2.0

    # Set default max_radius if not provided
    if max_radius is None:
        max_radius = (N / 2.0) - 1.0  # Subtract 1 to prevent going out of bounds

    # Calculate spiral parameter 'b' to ensure 'num_turns' within 'max_radius'
    b = max_radius / (num_turns * 2 * np.pi)

    # Create a grid of x and y coordinates relative to the center
    y_indices, x_indices = np.ogrid[0:N, 0:N]
    x = x_indices - center
    y = y_indices - center

    # Convert Cartesian coordinates to polar coordinates
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)  # Range: [-π, π)

    # Compute the ideal theta for the spiral at each radius
    theta_spiral = r / b

    # Compute the difference between the spiral theta and actual theta
    # Adjust for multiple turns by finding the closest multiple of 2π
    k = np.round((theta_spiral - theta) / (2 * np.pi))
    theta_adjusted = theta + 2 * np.pi * k
    delta_theta = np.abs(theta_spiral - theta_adjusted)

    # Convert thickness from pixels to radians
    # Since r = b * theta => dr = b * dtheta => dtheta = dr / b
    delta_theta_max = thickness / b

    # Create a mask where the angular difference is within the allowed thickness
    mask = delta_theta < delta_theta_max

    # Optionally, limit to max_radius to prevent wrapping beyond desired size
    mask &= r <= max_radius

    # Update the grid
    grid[mask] = 1

    return grid

star_array = generate_star_array(51, outer_radius=15, inner_radius=6.5)
spiral_array = generate_spiral_array(51, thickness=1.5, num_turns=2, max_radius=15)

M, N = 3, 51
u_stars = np.zeros((M,M,N,N))
u_stars[0,1] = shift(star_array, shift=(0,0))
u_stars[1,0] = shift(star_array, shift=(0,-10))
u_stars[1,2] = shift(star_array, shift=(0,10))
u_stars[2,1] = shift(spiral_array, shift=(10,10))

mask = np.zeros((M,M,N,N), dtype=bool)
mask[1,1,:,:] = True
u_stars_masked = np.ma.masked_array(u_stars, mask=mask)

########### GAUSSIAN AND GAUSSIAN-MIXTURE DISPLACEMENTS ###########

np.random.seed(42)

M_gaussians, N_gaussians = 5, 30
cov = np.array([[1,0],[0,12]])

def build_u_gaussians(M=M_gaussians, N=N_gaussians, cov=cov):
    means = np.zeros((M,M,2))
    thetas = np.zeros((M,M))
    gaussians = np.zeros((M, M, N, N))
    coord_0, coord_1 = 2*[ np.linspace(-8, 8, N) ]
    for i in range(M):
        for j in range(M):
            mean = 0 * np.array([np.random.rand(), np.random.rand()]) - 0
            theta = 1 * np.random.rand() - 0.5
            means[i,j] = mean
            thetas[i,j] = theta
            cov_temp = rotate_2d_array(cov, theta)
            gaussians[i,j] = shift(gaussian_function_as_matrix(coord_0, coord_1, cov=cov_temp), shift=mean )
    return gaussians

def build_u_gaussian_mixture(M=M_gaussians, N=N_gaussians, cov=cov):
    gaussian_mixture = build_u_gaussians(M=M, N=N)
    cov_rot_90 = rotate_2d_array(cov, np.pi/2)
    coord_0, coord_1 = 2*[ np.linspace(-8, 8, N) ]
    for i in range(M):
        for j in range(M):
            if np.random.rand() < 0.5: # decide how many voxels should be a mixture
                theta = 0.8 * np.random.rand() - 0.4
                cov_temp = rotate_2d_array(cov_rot_90, theta)
                gaussian_temp = gaussian_function_as_matrix(coord_0, coord_1, cov=cov_temp)
                gaussian_mixture[i,j] = gaussian_mixture[i,j] + gaussian_temp
    return gaussian_mixture.clip(min=0)

gaussian_mixture = build_u_gaussian_mixture()

###########

def plot_flow_fields(*arrays):
    """Visualizes multiple flow fields in the same plot using different colors."""
    if not arrays:
        raise ValueError("At least one flow field array is required.")
    
    N = arrays[0].shape[0]
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    
    plt.figure(figsize=(6, 6))

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']  # Different colors for multiple fields
    
    for i, A in enumerate(arrays):
        U = np.cos(A)  # X-component of the flow
        V = np.sin(A)  # Y-component of the flow
        plt.quiver(X, Y, U, V, scale=N, color=colors[i % len(colors)], alpha=0.6)

    plt.xlim(-0.5, N-0.5)
    plt.ylim(-0.5, N-0.5)
    plt.gca().set_aspect('equal')
    plt.title("Multiple Flow Fields Visualization")
    plt.show()

def generate_field(N, a, b, c=0):
    """Generates a (N,N) array of angles where the flow is symmetric 
    along the lower-left to upper-right diagonal."""
    X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))

    # Compute angles relative to the diagonal symmetry axis
    A = np.pi / 4 * (a*X + b*Y + c)  # Symmetric variation along the diagonal

    return A

def generate_field_2(N, a, c=0):
    """Generates a (N,N) array of angles representing a simple curved flow."""
    X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    
    # Simple curved flow: flow turns smoothly from left to right
    A = np.pi / 4 * ( a * np.exp(Y) + c )  # Linearly varying angle with Y
    
    return A

cov = np.array([[1,0],[0,20]])

def build_u_gaussians_fromfield(*fields, N=N_gaussians, cov=cov):
    M = len(fields[0])
    gaussians = np.zeros((M, M, N, N))
    coord_0, coord_1 = 2*[ np.linspace(-8, 8, N) ]
    for field in fields:
        for i in range(M):
            for j in range(M):
                # theta = 1 * np.random.rand() - 0.5
                cov_temp = rotate_2d_array(cov, field[i,j])
                gaussians[i,j] += gaussian_function_as_matrix(coord_0, coord_1, cov=cov_temp)
    return gaussians

def generate_simple_gaussian_crossing(M, N=N_gaussians):
    m = int(M/3)
    coord_0, coord_1 = 2*[ np.linspace(-8, 8, N) ]
    isotropic_diff = gaussian_function_as_matrix( coord_0, coord_1) * 0.01
    
    field_hor = np.ones((M,M)) * (-np.pi/2)
    horizontal = build_u_gaussians_fromfield(field_hor, N=N)
    
    indices = np.concatenate((np.arange(m), np.arange(len(horizontal) - m, len(horizontal))))
    horizontal[indices] = isotropic_diff
    field_vert = np.zeros((M,M))
    vertical = build_u_gaussians_fromfield(field_vert)
    vertical[:,indices] = isotropic_diff
    
    return horizontal + vertical

def create_4d_gaussians(covariances, N=20):
    """
    Creates a 4D array of 2D Gaussian density matrices.
    
    Parameters:
        covariances: A 4D array of shape (s0, s1, 2, 2), where each [i,j] is a 2x2 covariance matrix.
    
    Returns:
        A 4D array B of shape (s0, s1, N, N), where each B[i,j] is a 2D Gaussian density matrix.
    """
    coord_x = np.linspace(-3, 3, N)
    coord_y = np.linspace(-3, 3, N)
    
    # Get the shape of the covariances array
    s0, s1, _, _ = covariances.shape
    
    # Initialize the output array
    B = np.zeros((s0, s1, N, N))
    
    # Loop over each covariance matrix and compute the corresponding Gaussian density matrix
    for i in range(s0):
        for j in range(s1):
            cov = covariances[i, j]
            B[i, j] = gaussian_function_as_matrix(coord_x, coord_y, cov=cov)
    
    return B

field_1 = generate_field(11, 1, 1, 2)
field_2 = generate_field(11, 0, 0, -0.75)
flow_1 = build_u_gaussians_fromfield(field_1)
flow_2 = build_u_gaussians_fromfield(field_2)
flow_gaussians_crossing = flow_1 + flow_2
    

########### SEBASTIAN'S TENSOR ###########

def import_sebastian(whichslice=3, normalize_entries=True):
    from scipy.io import loadmat
    dt_slice = loadmat('dt.mat')['dt'][:,:,whichslice]
    mask = loadmat('dt.mat')['mask'][:,:,whichslice]
    m, n = mask.shape # m, n = 120
    dt_slice_matrices = np.zeros((m,n,3,3))
    for i in range(m):
        for j in range(n):
            Dxx, Dyy, Dzz, Dyz, Dxz, Dxy = dt_slice[i,j,:]
            dt_slice_matrices[i,j][np.triu_indices(3)] = [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
            dt_slice_matrices[i,j] += np.triu(dt_slice_matrices[i,j], k=1).T
            if ~np.all(np.linalg.eigvals(dt_slice_matrices[i,j]) >= 0):
                mask[i,j] = False
                dt_slice_matrices[i,j] = 0
    if normalize_entries: dt_slice_matrices = dt_slice_matrices/np.max(dt_slice_matrices)
    return dt_slice_matrices, np.array(mask, dtype=bool)

def build_sebastian_distributions(N=21, normalize_tensors = False):
    dt, mask = import_sebastian()
    if normalize_tensors:
        minimum, maximum = np.min(dt), np.max(dt)
        dt = dt / maximum
    M = dt.shape[0]
    coord = np.linspace(-1.5,1.5,N)
    distributions = np.zeros((M,M,N,N,N))
    for i in range(M):
        for j in range(M):
            if mask[i,j]:
                distributions[i,j] = gaussian_function_as_matrix(coord,coord,coord, cov=dt[i,j])
    return distributions, mask

###### Forward operator: Fourier + Undersampling #####

def UF(x, mask=None):
    return np.ma.masked_array(np.fft.fftn(x, norm='ortho'), mask=mask)

def UFstar(y):
    return np.fft.ifftn(np.ma.filled(y, fill_value=0), norm='ortho')
