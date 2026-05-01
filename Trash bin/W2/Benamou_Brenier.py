import numpy as np
import scipy.sparse as sp
from algorithms_general import CP
from differential_operators_sparse import create_partialplus_sparse, create_staggered_partialplus_sparse
from prox_and_proj import prox_square_over_linear

def build_K_Benamou_Brenier(f, g, M):
    
    fx, fy = f.shape
    shape_rho = (M,) + (fx , fy)
    shape_mx = (M-1,) + (fx-1 , fy)
    shape_my = (M-1,) + (fx , fy-1)

    shape_mx_dual = (M-1,) + (fx, fy)
    
    len_rho = np.prod(shape_rho)
    len_mx = np.prod(shape_mx)
    len_my = np.prod(shape_my)
    
    d_t = create_partialplus_sparse(shape_rho, 0, boundary='no')
    d_x = create_partialplus_sparse(shape_mx_dual, 1, boundary='no')
    d_y = create_partialplus_sparse(shape_mx_dual, 2, boundary='no')
    
    div_CE = sp.bmat( [ [ d_t , d_x.T , d_y.T ] ], format='csr' )
    
    where_rho = np.zeros(shape_rho) # to select the boundaries
    where_rho[[0,-1],:,:] = 1
    where_rho = where_rho.ravel()
    selected_ind_rho = np.where( where_rho == 1 )[0] # np.where returns ( [^list of indices^], ^type^ )
    rows_rho = np.arange(len(selected_ind_rho))  # Row indices (0 to num_selected - 1)
    cols_rho = selected_ind_rho
    data_rho = np.ones(len(selected_ind_rho))
    b_rho = sp.csr_matrix((data_rho, (rows_rho, cols_rho)), shape=(len(selected_ind_rho), np.prod(shape_rho) ))
    # b_rho è tale che b_rho @ rho.ravel() seleziona le entrate di rho sui bordi
    
    select_edges = sp.bmat( [ [ b_rho, np.zeros( ( b_rho.shape[0], len_mx + len_my ) ) ] ], format="csr")
    K = sp.vstack([div_CE, select_edges], format="csr")
    
    # b0 = np.zeros(K.shape[0])
    # b0_tilde = np.zeros( select_edges.shape[0] )
    # fg_ravelled = np.stack((f,g), axis=0).ravel()
    # b0_tilde[ :b_rho.shape[0] ] = fg_ravelled
    
    b0 = np.zeros(K.shape[0])
    fg_ravelled = np.concatenate( ( f.ravel(), g.ravel() ) )
    b0[ -b_rho.shape[0]: ] = fg_ravelled
    
    shapes = (shape_rho, shape_mx, shape_my)
    
    return K, b0, shapes #, select_edges, b_rho, b_mx, b_my, d_t, d_x, d_y, div_CE

def sparse_affine_proj(K, b, x0): # projection on the affine set { x : K x = b }
    return x0 - K.T @ sp.linalg.spsolve( K @ K.T , K @ x0 - b )

def padding_matrix(shape, axis, pad_width=1):
    shape_new = list(shape)
    shape_new[axis] += pad_width

    size_old = np.prod(shape)
    size_new = np.prod(shape_new)

    # Generate indices of the original array
    idxs = np.arange(size_old)

    # Convert to multi-indices
    multi_idxs = np.unravel_index(idxs, shape)
    multi_idxs = np.stack(multi_idxs, axis=-1)  # shape: (size_old, ndim)

    # Compute new flattened indices in the padded array
    new_multi_idxs = multi_idxs.copy()
    new_flat_idxs = np.ravel_multi_index(new_multi_idxs.T, shape_new)

    # Sparse matrix that maps old flattened array to the padded version
    A = sp.csr_matrix((np.ones(size_old), (new_flat_idxs, np.arange(size_old))), shape=(size_new, size_old))
    return A

def W22(f, g, M, sig, maxit=1000):
    
    from prox_and_proj import prox_square_over_linear
    from algorithms_general import pDR_Richardson as pDR
    from averaging_arrays import create_avg_sparse_zeroboundary
    
    K, b0, shapes = build_K_Benamou_Brenier(f, g, M)
    shape_rho, shape_mx, shape_my = shapes
    len_rho, len_mx, len_my = np.prod(shape_rho), np.prod(shape_mx), np.prod(shape_my)
    
    rho, mx, my = np.zeros(len_rho), np.zeros(len_mx), np.zeros(len_my) # variables are defined flat
    
    xi0 = { 'rho': rho,
            'mx': mx,
            'my': my }
    
    # !!! These matrices are heavy to create and should't be created each time if W22
    # is called many time (on a (30,30,30) shape it takes 18 secs)
    
    A_t = np.zeros( ( np.prod(np.array(shape_rho) - np.array((1,0,0))), len_rho ) )
    np.fill_diagonal(A_t, 1)
    A_x = create_avg_sparse_zeroboundary(shape_mx, 1)
    A_y = create_avg_sparse_zeroboundary(shape_my, 2)
    
    def prox_f(xi, sig): # projection on the constraint K @ xi = b
        rho, mx, my = xi.values()
        out = sparse_affine_proj( K, b0, np.concatenate( tuple(xi.values()) ) )
        return { 'rho': out[:len(rho)],
                  'mx': out[len(rho):-len(my)],
                  'my': out[-len(my):] }
    
    def prox_g(A_xi, sig): # prox of \mathcal{J}
        rho, mx, my = A_xi.values()  
        m = np.vstack( (mx, my) ).T # now m has shape (*,2)
        m, rho = prox_square_over_linear(m, rho, step=sig)
        return { 'A_rho': rho, 'A_mx': m[:,0], 'A_my': m[:,1] }
    
    def L(xi):
        rho, mx, my = xi.values() 
        return { 'A_rho': A_t @ rho ,
                 'A_mx': A_x @ mx,
                 'A_my': A_y @ my }
    def Lstar(A_xi):
        rho, mx, my = A_xi.values()
        return { 'rho': A_t.T @ rho ,
                 'mx': A_x.T @ mx,
                 'my': A_y.T @ my }
    
    sing_val_t = sp.linalg.svds(A_t, k=1, return_singular_vectors=False)[0]
    sing_val_x = sp.linalg.svds(A_x, k=1, return_singular_vectors=False)[0]
    sing_val_y = sp.linalg.svds(A_y, k=1, return_singular_vectors=False)[0]
    norm_L_sq = max( sing_val_t, sing_val_x, sing_val_y )
    print(norm_L_sq)
    
    X, S = pDR(xi0, sig, prox_f, prox_g, L, Lstar, norm_L_sq, iterations=maxit, printprogress=True)
    
    return X

def W22_tentantivo_con_average_su_rho(f, g, M, sig, maxit=1000):
    
    from prox_and_proj import prox_square_over_linear
    from algorithms_general import pDR_Richardson as pDR
    
    K, b0, shapes = build_K_Benamou_Brenier(f, g, M)
    shape_rho, shape_mx, shape_my = shapes
    len_rho, len_mx, len_my = np.prod(shape_rho), np.prod(shape_mx), np.prod(shape_my)
    
    rho, mx, my = np.zeros(len_rho), np.zeros(len_mx), np.zeros(len_my) # variables are defined flat
    
    xi0 = { 'rho': rho,
            'mx': mx,
            'my': my }
    
    # !!! These matrices are heavy to create and should't be created each time if W22
    # is called many time (on a (30,30,30) shape it takes 18 secs)
    
    # A_mx = padding_matrix(shape_mx, 1)
    # A_my = padding_matrix(shape_my, 2)
    
    # A_rho_cut cuts the last part of rho (corresponding to "g")
    shape_rho_shortened = np.array(shape_rho) - np.array((1,0,0))
    size = min(np.prod(shape_rho_shortened), len_rho)
    # A_rho_cut = sp.diags([1.0] * size, offsets=0, shape=(np.prod(shape_rho_shortened), len_rho), format='csr')
    
    from averaging_arrays import average_along_axis_as_matrix
    
    A_t = average_along_axis_as_matrix(shape_rho, 0)
    A_x = average_along_axis_as_matrix(shape_rho_shortened, 1) @ A_t # this product is valide because the truncated ("cut") part comes last in the flattening
    A_y = average_along_axis_as_matrix(shape_rho_shortened, 2) @ A_t
    
    def prox_f(xi, sig): # projection on the constraint K @ xi = b
        rho, mx, my = xi.values()
        out = sparse_affine_proj( K, b0, np.concatenate( tuple(xi.values()) ) )
        return { 'rho': out[:len(rho)],
                  'mx': out[len(rho):-len(my)],
                  'my': out[-len(my):] }
    
    def prox_g(z, sig): # prox of \mathcal{J}
        rho_x, rho_y, mx, my = z.values()
        mx = mx.reshape((mx.shape[0], -1))
        my = my.reshape(my.shape[0], -1)
        mx, rho_x = prox_square_over_linear(mx, rho_x, step=sig)
        my, rho_y = prox_square_over_linear(my, rho_y, step=sig)
        return { 'rho_x': rho_x, 'rho_y': rho_y, 'mx': mx[:,0], 'my': my[:,0] }
    
    def L(xi):
        rho, mx, my = xi.values() 
        return { 'rho_x': A_x @ rho,
                 'rho_y': A_y @ rho,
                 'mx': mx,
                 'my': my }
    def Lstar(z):
        rho_x, rho_y, mx, my = z.values()
        return { 'rho': A_x.T @ rho_x + A_y.T @ rho_y ,
                 'mx': mx,
                 'my': my }
    
    sing_val = sp.linalg.svds( A_x.T @ A_x + A_y.T @ A_y, k=1, return_singular_vectors=False )[0]
    norm_L_sq = max( sing_val, 1 )
    print(norm_L_sq)
    
    X, S = pDR(xi0, sig, prox_f, prox_g, L, Lstar, norm_L_sq, iterations=maxit, printprogress=True)
    
    return X

def W22_3(f, g, M=20, maxit=5000): # old implementation

    from differential_operators import partialplus, partialminus, nabla_y, div_y    

    maxval = max( np.max(f), np.max(g) )
    def proj_C_rho(rho):
        rho1 = rho.clip(min=0).clip(max=maxval)
        rho1[0], rho1[-1] = f, g
        return rho1
    def proj_C_m(m):
        m1 = m.copy()
        m1[0], m1[-1] = 0, 0
        return m1
    
    def prox_f(X, tau):
        m_tmp, rho_tmp = prox_square_over_linear( X['m'].reshape(-1, X['m'].shape[-1]), X['rho'].ravel(), tau )        
        return { 'rho': proj_C_rho( rho_tmp.reshape(X['rho'].shape) ) ,
                 'm':   proj_C_m( m_tmp.reshape(X['m'].shape) ) }
    
    prox_gstar = lambda Y, sig: Y
    
    L = lambda X: { 'p': partialplus(X['rho'], 0) + div_y(X['m'], axis=-1) }
    Lstar = lambda Y: { 'rho': - partialminus(Y['p'], 0) ,
                        'm': - np.moveaxis( nabla_y(Y['p'], dim=2), 0, -1) }
    
    tau, sig = 2e-1/np.sqrt(12), 1e1/(2*np.sqrt(12))
    
    shape_rho = (M,) + f.shape
    

    X0 = { 'rho': rho0 ,
           'm': m0 }
    
    p0 = partialminus(rho0, 0)
    Y0 = { 'p': p0 }
    
    X, Y = CP(X0, Y0, tau, sig, prox_f, prox_gstar, L, Lstar, maxit, printprogress=True)
    
    return X, Y

# M = 20
# fx, fy = 10, 10
# f = np.zeros((fx,fy))
# g = f.copy()

# f[0,0] = 1
# g[-1,-1] = 1

# K, b0, shapes = build_K_Benamou_Brenier(f, g, M=M)

from PIL import Image

def read_image(img1, p):
    img_brg = Image.open(img1).convert('L')
    Img = 255-np.array(img_brg.resize((p, p)))
    Img = np.reshape(Img, p**2)
    return Img

def read_measures(img1, img2, p):
    mu = read_image(img1, p)
    mu = mu/np.sum(mu)
    nu = read_image(img2, p)
    nu = nu/np.sum(nu)
    return mu, nu

p=20
[mu, nu] = read_measures("Data/mu.png", "Data/nu.png", p=p)
mu, nu = mu.reshape(p,p), nu.reshape(p,p)

# # To plot rho and m side to side

# M = 10
# sig = 1
# # X = W22(mu, nu, M, sig, maxit=5000)
# # rho, mx, my = X.values()
# shape_rho = (M,) + mu.shape
# rho = rho.reshape(shape_rho)

# shape_mx = (M-1, mu.shape[0]-1, mu.shape[1] )
# shape_my = (M-1, mu.shape[0], mu.shape[1]-1 )

# shape_after_A = (M-1, mu.shape[0], mu.shape[1] )

# mx = mx.reshape(shape_mx)
# my = my.reshape(shape_my)

# A_t = np.zeros( ( np.prod(np.array(shape_rho) - np.array((1,0,0))), np.prod(shape_rho) ) )
# np.fill_diagonal(A_t, 1)
# A_x = create_avg_sparse_zeroboundary(shape_mx, 1)
# A_y = create_avg_sparse_zeroboundary(shape_my, 2)

# mx_avg = ( A_x @ mx.ravel() ).reshape(shape_after_A)
# my_avg = ( A_y @ my.ravel() ).reshape(shape_after_A)

# m = np.stack( (mx_avg, my_avg), axis=0 )
# m_moveax = np.moveaxis(m, 0, -1)


# for i in range(len(mx_avg)):

#     fig, axes = plt.subplots( 1, 2, figsize=(12,6) )

#     m, n = mx_avg[i].shape
#     x = np.arange(n)
#     y = np.arange(m)
#     X, Y = np.meshgrid(x, y)
    
#     axes[0].imshow(rho[i], cmap='gray', vmax=np.max(mu))
#     axes[1].quiver(X, Y, mx_avg[i], my_avg[i])
#     axes[1].invert_yaxis()
#     plt.show()
    
#     print(i, ':, ', 100*np.linalg.norm(m_moveax[i, 1,10]) )
