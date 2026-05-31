import numpy as np


def L2_TVPR(b, alpha1, eps, beta1 = None, forward='UF', maxit=10000, printprogress=True):
    
    from prox_and_proj import prox_norm21, proj_L_infty_ball
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from algorithms_general import CP
    
    b = np.ma.masked_array(b) # ensures b is a masked array
    
    dim_x = {3: 1, 4: 2, 5: 2}.get(b.ndim, None) # I am allowing 3-, 4- and 5-D data
    dim_y = b.ndim - dim_x
    L_norm_sq = 2 + 4 * (dim_x + 2*dim_y)
    sig, tau = 2 * [1 / np.sqrt(L_norm_sq)] # this is to have sig = tau
    if beta1 in {None}: beta1 = alpha1 * np.max(b.shape[dim_x+1:])
    
    if forward == False: b = np.ma.masked_array(b, mask=np.zeros(b.shape, dtype=bool)) # I ensure that b is a masked array even if forward is False
    
    U     = lambda x: np.ma.masked_array(x, mask=b.mask)
    Ustar = lambda lam: lam.filled(0)
    
    if forward in {'UF'}:
        K     = lambda u: U( np.fft.fftn(u, norm='ortho') )
        Kstar = lambda lam: np.fft.ifftn( Ustar(lam) , norm='ortho')
    elif forward in {'U'}:
        K     = U
        Kstar = Ustar
    elif forward in {False}:
        K, Kstar = [ lambda x: x.flatten(), lambda x: x.reshape(b.shape) ]
        
    prox_f     = lambda X, tau: { 'u':   np.real(X['u']),
                                  'xi':  prox_norm21(X['xi'], lam=alpha1*tau) ,
                                  'gam': prox_norm21(X['gam'], lam=eps*tau) }
    prox_gstar = lambda Y, sig: { 'lam': ( Y['lam'] - sig*b ) / (1 + sig) ,
                                  'w':   proj_L_infty_ball(Y['w'], lam=beta1) ,
                                  'eta': Y['eta'] }
    L     = lambda X: { 'lam': K(X['u']) ,
                        'w':   nabla_x(X['u'], dim=dim_x) + div_y(X['xi']) ,
                        'eta': nabla_y(X['u'], dim=dim_y) - X['gam'] }
    Lstar = lambda Y: { 'u':   Kstar(Y['lam']) - div_x(Y['w']) - div_y(Y['eta']) ,
                        'xi':  -nabla_y(Y['w']) ,
                        'gam': -Y['eta'] }
    temp0 = np.zeros(b.shape)
    X0 = { 'u':   temp0 ,
           'xi':  nabla_y( nabla_x(temp0, dim=dim_x), dim=dim_y) ,
           'gam': nabla_y(temp0, dim=dim_y) }
    Y0 = { 'lam': K(temp0) ,
           'w':   nabla_x(temp0, dim=dim_x) ,
           'eta': nabla_y(temp0, dim=dim_y) }
    
    X, Y = CP(X0, Y0, tau, sig, prox_f, prox_gstar, L, Lstar, maxit, printprogress=printprogress)
    
    return X['u'].clip(min=0), Y['w']

def L2_TVPR_Dirichlet(b, alpha1, eps, beta1 = None, forward='UF', algorithm='CP' , maxit=10000, printprogress=True):
    
    from prox_and_proj import prox_norm21, proj_L_infty_ball
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from algorithms_general import CP, pDR_Richardson
    
    b = np.ma.masked_array(b) # ensures b is a masked array
    
    dim_x = {3: 1, 4: 2, 5: 2}.get(b.ndim, None) # I am allowing 3-, 4- and 5-D data
    dim_y = b.ndim - dim_x
    L_norm_sq = 1 + 4 * (dim_x + (1+eps)*dim_y)
    
    if beta1 in {None}: beta1 = alpha1 * np.max(b.shape[dim_x+1:])
    
    if forward == False: b = np.ma.masked_array(b, mask=np.zeros(b.shape, dtype=bool)) # I ensure that b is a masked array even if forward is False
    
    U     = lambda x: np.ma.masked_array(x, mask=b.mask)
    Ustar = lambda lam: lam.filled(0)
    
    if forward in {'UF'}:
        K     = lambda u: U( np.fft.fftn(u, norm='ortho') )
        Kstar = lambda lam: np.fft.ifftn( Ustar(lam) , norm='ortho')
    elif forward in {'U'}:
        K     = U
        Kstar = Ustar
    elif forward in {False}:
        K, Kstar = [ lambda x: x.flatten(), lambda x: x.reshape(b.shape) ]
        
    prox_f     = lambda X, tau: { 'u':   np.real(X['u']),
                                  'xi':  prox_norm21(X['xi'], lam=alpha1*tau, ax=1) }
    prox_gstar = lambda Y, sig: { 'lam': ( Y['lam'] - sig*b ) / (1 + sig) ,
                                  'w':   proj_L_infty_ball(Y['w'], lam=beta1) ,
                                  'eta': Y['eta'] / (1+sig*eps) }
    L     = lambda X: { 'lam': K(X['u']) ,
                        'w':   nabla_x(X['u'], dim=dim_x) + div_y(X['xi']) ,
                        'eta': eps * nabla_y(X['u'], dim=dim_y) }
    Lstar = lambda Y: { 'u':   Kstar(Y['lam']) - div_x(Y['w']) - eps * div_y(Y['eta']) ,
                        'xi':  -nabla_y(Y['w'], dim=dim_y) }
    temp0 = np.zeros(b.shape)
    X0 = { 'u':   temp0 ,
           'xi':  nabla_y( nabla_x(temp0, dim=dim_x), dim=dim_y) }
    Y0 = { 'lam': K(temp0) ,
           'w':   nabla_x(temp0, dim=dim_x) ,
           'eta': nabla_y(temp0, dim=dim_y) }
    
    if algorithm == 'CP':
        sig, tau = 2 * [1 / np.sqrt(L_norm_sq)] # this is to have sig = tau
        param_factor = 1000 # for best convergence tau is 10^6 times sigma!
        tau = param_factor / L_norm_sq
        sigma = tau / param_factor
        X, Y = CP(X0, Y0, tau, sig, prox_f, prox_gstar, L, Lstar, maxit, printprogress=printprogress)
    
    if algorithm == 'DR':
        sig = 0.5 + 0.5 * (1 / L_norm_sq) # Bredies uses sig=1, but often 1 / L_norm_sq is chosen 
        X, Y = pDR_Richardson( X0, sig, prox_f, prox_gstar, L, Lstar, L_norm_sq, iterations=maxit, printprogress=printprogress )
    
    return X['u'].real.clip(min=0), Y['w']

def L2_2(b, maxit=4000, printprogress=True): # funziona!
    
    from algorithms_general import CP
    
    data_full, mask = b.data, np.invert(b.mask)
    data = data_full[mask]
    
    L_norm_sq = 1
    sig, tau = 2 * [1 / np.sqrt(L_norm_sq)] # this is to have sig = tau
    
    U     = lambda x: x[mask]
    def Ustar(lam):
        x = np.zeros_like(data_full)
        x[mask] = lam
        return x

    K     = lambda u: U( np.fft.fftn(u, norm='ortho') )
    Kstar = lambda lam: np.fft.ifftn( Ustar(lam) , norm='ortho')
    
    prox_f     = lambda X, tau: { 'u':   np.real(X['u']) }
    prox_gstar = lambda Y, sig: { 'lam': ( Y['lam'] - sig*data ) / (1 + sig) }
    L     = lambda X: { 'lam': K(X['u']) }
    Lstar = lambda Y: { 'u':   Kstar(Y['lam']) }
    
    temp0 = np.zeros_like(data_full, dtype=float)
    X0 = { 'u':   temp0 }
    Y0 = { 'lam': K(temp0) }
    
    X, Y = CP(X0, Y0, tau, sig, prox_f, prox_gstar, L, Lstar, maxit, printprogress=printprogress)
    
    return X['u'].real

def L2(b, maxit=4000, printprogress=True): # funziona!
    
    from algorithms_general import CP
    
    b = np.ma.masked_array(b) # ensures b is a masked array
    
    L_norm_sq = 1
    sig, tau = 2 * [1 / np.sqrt(L_norm_sq)] # this is to have sig = tau
    
    U     = lambda x: np.ma.masked_array(x, mask=b.mask)
    Ustar = lambda lam: lam.filled(0)

    K     = lambda u: U( np.fft.fftn(u, norm='ortho') )
    Kstar = lambda lam: np.fft.ifftn( Ustar(lam) , norm='ortho')
    
    # prova_1 = np.random.rand(5,5,30,30) + 1j*np.random.rand(5,5,30,30)
    # prova_2 = np.ma.masked_array( np.random.rand(5,5,30,30) + 1j*np.random.rand(5,5,30,30), mask=b.mask )
    # print( np.vdot( K(prova_1).compressed() , prova_2.compressed()) )
    # print( np.vdot( prova_1 , Kstar(prova_2) ) )
    
    prox_f     = lambda X, tau: { 'u':   np.real(X['u']) }
    prox_gstar = lambda Y, sig: { 'lam': ( Y['lam'] - sig*b ) / (1 + sig) }
    L     = lambda X: { 'lam': K(X['u']) }
    Lstar = lambda Y: { 'u':   Kstar(Y['lam']) }
    
    temp0 = np.zeros(b.shape, dtype=float)
    X0 = { 'u':   temp0 }
    Y0 = { 'lam': K(temp0) }
    
    X, Y = CP(X0, Y0, tau, sig, prox_f, prox_gstar, L, Lstar, maxit, printprogress=printprogress)
    
    return X['u'].real

def TVPR( u, alpha1, tau, beta1=None , maxit=2000, printprogress=False ):
    '''
    
    Evaluate the TV-PR (Piccoli-Rossi) regularizer of a field u, via CP on its
    dual. Returns (s1, s2): s1 = ||xi||_{2,1} (the regularizer value, 2-norm
    grouped along the displacement-gradient axis 0); s2 = <nabla_x(u/alpha1) +
    div_y(xi), w>, a duality check that should match s1 at convergence.
    prox_norm21 below now passes ax=0 (it was omitted, relying on a removed
    default of 1) to match s1 and the canonical naif model.
    
    '''
    from prox_and_proj import prox_norm21, proj_L_infty_ball
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from algorithms_general import CP
    
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
    