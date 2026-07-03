import numpy as np
import ot

def TVW1(u, shape_x, shape_y, ground_metric=None, reg=None, iso=False):
    """
    Total Variation Wasserstein-1 seminorm.

    Sums the W1 distances between every pair of spatially adjacent slices of u
    along each dimension of shape_x.

    Parameters
    ----------
    u : array-like, shape shape_x + shape_y
        Each "pixel" u[x] is interpreted as a (non-negative) measure over the
        shape_y grid.
    shape_x : tuple of int
        Spatial (grid) dimensions.
    shape_y : tuple of int
        Support (distribution) dimensions.
    ground_metric : (prod(shape_y), prod(shape_y)) array, optional
        Ground cost matrix. Defaults to pairwise Euclidean distances on the
        shape_y integer grid.
    reg : float, optional
        Entropy regularization. None → exact OT (ot.emd2); float → Sinkhorn
        (ot.sinkhorn2), which is differentiable.
    iso : bool
        Isotropic variant (not yet implemented).

    Returns
    -------
    float
    """
    if iso:
        raise NotImplementedError("Isotropic TVW1 is not implemented yet.")

    u = np.asarray(u, dtype=float)
    assert u.shape == tuple(shape_x) + tuple(shape_y), (
        f"u.shape {u.shape} != shape_x + shape_y {tuple(shape_x) + tuple(shape_y)}"
    )

    n_x = len(shape_x)
    ny = int(np.prod(shape_y))

    if ground_metric is None:
        # Integer grid coordinates for shape_y, then pairwise Euclidean cost
        coords = np.array(list(np.ndindex(*shape_y)), dtype=float)
        ground_metric = ot.dist(coords, coords, metric='euclidean')

    M = np.ascontiguousarray(ground_metric, dtype=np.float64)
    total = 0.0
    sl_base = [slice(None)] * u.ndim

    for d in range(n_x):
        sl_a = sl_base.copy()
        sl_b = sl_base.copy()
        sl_a[d] = slice(None, -1)
        sl_b[d] = slice(1, None)

        a_batch = u[tuple(sl_a)]  # shape: (modified shape_x) + shape_y
        b_batch = u[tuple(sl_b)]

        n_pairs = int(np.prod(a_batch.shape[:n_x]))
        a_flat = a_batch.reshape(n_pairs, ny)  # (n_pairs, ny)
        b_flat = b_batch.reshape(n_pairs, ny)

        for i in range(n_pairs):
            ai = np.ascontiguousarray(a_flat[i], dtype=np.float64)
            bi = np.ascontiguousarray(b_flat[i], dtype=np.float64)
            # Kantorovich-Rubinstein: W1(a,b) = emd2((a-b)+, (a-b)-, M)
            # Works for probability measures AND zero-sum signed measures.
            # Requires ai.sum() == bi.sum() (balanced); raises if not.
            diff = ai - bi
            pos = np.maximum(diff, 0.0)
            neg = np.maximum(-diff, 0.0)
            mass = pos.sum()
            if mass < 1e-15:
                continue  # ai ≈ bi — W1 = 0
            if reg is None:
                total += float(ot.emd2(pos, neg, M))
            else:
                total += float(ot.sinkhorn2(pos, neg, M, reg)[0])

    return total

def TVPR( u, alpha1, tau, beta1=None , maxit=2000, printprogress=False ):# !! Extremely slow to converge?
    '''
    
    Evaluate the TV-PR (Piccoli-Rossi) regularizer of a field u, via CP on its
    dual. Returns (s1, s2): s1 = ||xi||_{2,1} (the regularizer value, 2-norm
    grouped along the displacement-gradient axis 0); s2 = <nabla_x(u/alpha1) +
    div_y(xi), w>.
    
    '''
    from prox_and_proj import prox_norm21, proj_L_infty_ball
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from proximal_algorithms.CP import CP
    
    dim_x = {3: 1, 4: 2, 5: 2}.get(u.ndim, None) # I am allowing 3-, 4- and 5-D data
    dim_y = u.ndim - dim_x
        
    L_norm_sq = 4 * dim_y
    sig = 1 / (tau * L_norm_sq)
    # sig, tau = 2 * [1 / np.sqrt(L_norm_sq)] # this is to have sig = tau
    if beta1 in {None}: beta1 = alpha1 * np.max(u.shape[dim_x+1:])
    
    prox_f     = lambda X, tau: { 'xi':  prox_norm21(X['xi'], lam=tau, ax=0) }
    prox_gstar = lambda Y, sig: { 'w':   proj_L_infty_ball(Y['w'] + sig/alpha1 * nabla_x(u, dim=dim_x), lam=beta1) }
    L     = lambda X: { 'w':   div_y(X['xi']) }
    Lstar = lambda Y: { 'xi':  -nabla_y(Y['w'], dim=dim_y) }
    
    temp0 = u.copy()
    X0 = { 'xi':  nabla_y( nabla_x(temp0, dim=dim_x), dim=dim_y) }
    Y0 = { 'w':   nabla_x(temp0, dim=dim_x) }
    
    X, Y = CP(X0, Y0, tau, sig, prox_f, prox_gstar, L, Lstar, maxit, printprogress=printprogress)
    
    # if np.max(np.abs(Y['w'])) > beta1:
    #     w_proj = proj_L_infty_ball(Y['w'], lam=beta1)
    #     print('warning: w did not belong to the infinity ball')
    
    s1 = np.sum( np.linalg.norm( X['xi'], axis=0 ) )
    s2 = np.dot( nabla_x( u/alpha1, dim=dim_x).flatten() + div_y(X['xi']).flatten() , Y['w'].flatten() )
    
    return s1, s2

if __name__ == '__main__':

    sh_x, sh_y = (4,4), (9,9)
    ndim_y = len(sh_y)
    tmp = np.zeros(sh_x + sh_y)
    tmp[:,:,1,1] = 1
    tmp[1,3,1,1] = 0
    tmp[1,3,1,2] = 1

    from plottings import plot_u
    plot_u(tmp, ndim_y)

    dist = TVW1(tmp, sh_x, sh_y)

    print(dist)

    tmp2 = np.zeros(sh_x + sh_y)
    tmp2[:,:,1,1] = 1
    tmp2[:,:,3,3] = -1
    tmp2[1,3,1,1] = 0
    tmp2[1,3,1,2] = 1

    plot_u(tmp2, ndim_y)