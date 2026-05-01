import numpy as np

alpha0, beta0 = 1, 1*40 # Fidelity: Lipschitz / infinity (alpha0=np.inf means L1)
alpha1, beta1 = 0.1, 0.1*40 # TV: Lipschitz / infinity (alpha1=np.inf means TVL1)
par_PR_TVPR = np.array([alpha0, alpha1, beta0, beta1]) # Note: 0 and np.inf values are allowed

alpha0, beta0 = 1, 18
beta1 = 4
par_PR_TVL1 = [alpha0, beta0, beta1]

beta0, beta1 = 1, 1
par_L1_TVL1 = [beta0, beta1]

alpha1, beta1 = 1, 1*30
par_Inpainting_TVPR = [alpha1, beta1]

alpha1, beta1 = 0, 0*30
par_L2_TVPR = [alpha1, beta1]

eps = 0 # sparsity parameter
tau = 0.25 # first stepsize

def L2_TVPR_old(b, forward, parameters=par_L2_TVPR, initial_data='zero', iterations=10000, method='CP', tau=tau, eps=eps):
    
    import numpy as np
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from prox_and_proj import prox_norm21, proj_L_infty_ball, proj_simplex_array
    from algorithms_general import CP
    
    alpha1, beta1 = parameters
    
    b = np.ma.masked_array(b) # if b is not a masked.array, I define it as one
    
    def UF(x): return np.ma.masked_array(np.fft.fftn(x, norm='ortho'), mask=b.mask) # I define the forward operator
    def UFstar(y): return np.fft.ifftn(np.ma.filled(y, fill_value=0), norm='ortho')
    if forward in {'UF'}: K, Kstar = UF, UFstar
    elif forward in {False}: K, Kstar = 2 * [lambda x: x] # I also allow for no forward operator
    elif forward in {'U'}: K, Kstar = lambda x: np.ma.masked_array(x, mask=b.mask), lambda x: np.ma.filled(x, fill_value=0)
    elif forward in {'F', 'fft'}: K, Kstar = lambda x: np.fft.fftn(x, norm='ortho'), lambda x: np.fft.ifftn(x, norm='ortho')
    else: raise Exception('Parameter '"forward"' must be True or False')
    
    if b.ndim == 5: dim_x, Lnorm_sq = 3, 21
    elif b.ndim == 4: dim_x, Lnorm_sq = 2, 17
    elif b.ndim == 3: dim_x, Lnorm_sq = 1, 13
    else: raise Exception('b.ndim is neither 3 nor 4 nor 5')
    
    sig = 1 / (tau * Lnorm_sq) # second stepsize
    def L(X):
        return {'lam': K(X['u']) ,
                'w': nabla_x(X['u'], dim_x) + div_y(X['xi']) }

    def Lstar(Y):
        return {'u': Kstar(Y['lam']) - div_x(Y['w']) ,
                'xi': -nabla_y(Y['w']) }

    def prox_F(X, tau):
        return {'u': proj_simplex_array(X['u'] - tau*eps, 2) ,
                'xi': prox_norm21(X['xi'], lam=alpha1*tau) }

    def prox_Gstar(Y, sig):
        return {'lam': (Y['lam'] - sig*b) / (1+sig) , # a difference of ma.arrays only acts on visible elements
                'w': proj_L_infty_ball(Y['w'], lam=beta1) }
    
    if isinstance(initial_data, np.ndarray): u0 = initial_data
    elif initial_data in {'random', 'rand'}: u0 = np.random.rand(*b.shape)
    elif initial_data in {'zero', 'zeros'}: u0 = np.zeros(b.shape)
    elif initial_data in {'b'}: u0 =  np.real(Kstar(b))
    w0 = nabla_x( u0, dim_x)
    xi0 =  nabla_y(w0)
    lam0 = np.ma.masked_array(np.zeros(b.shape), mask=b.mask)
    
    
    X = {'u': u0, 'xi': xi0}
    Y = {'lam': lam0, 'w': w0}
    if method == 'CP':
        X, Y = CP(X, Y, tau, sig, prox_F, prox_Gstar, L, Lstar, iterations)
        return X['u'], Y['w']
    else: print('Method argument not recognised')
    return

def PR_TVPR_forward(b, alpha1=0.1, initial_data='zero', forward='fft', iterations=10000, method='CP', tau=tau, eps=eps):
    
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from prox_and_proj import prox_norm21, proj_L_infty_ball
    from algorithms_general import CP
    
    def UF(x, mask=None): return np.ma.masked_array(np.fft.fftn(x, norm='ortho'), mask=mask)
    def UFstar(y): return np.fft.ifftn(np.ma.filled(y, fill_value=0), norm='ortho')
    
    if b.ndim == 5: dim_x, Lnorm_sq = 3, 34
    elif b.ndim == 4: dim_x, Lnorm_sq = 2, 26
    elif b.ndim == 3: dim_x, Lnorm_sq = 1, 22
    else: raise Exception('b.ndim is neither 3 nor 4 nor 5')
    dim_y = b.ndim - dim_x
    
    sig = 1 / (tau * Lnorm_sq) # second stepsize
    
    mask = getattr(b, 'mask', None)
    
    if forward == None:
        K = lambda x: x
        Kstar = lambda x: x
    if forward in {'Fourier', 'fft', 'FFT'}:
        K = lambda x: UF(x, mask)
        Kstar = UFstar

    if initial_data in {'rand', 'random'}: u0 = np.random.rand(*b.shape)
    if initial_data in {0, 'zero', 'zeros'}: u0 = np.zeros(b.shape)
    proj = lambda x : x + Kstar(b - K(x))
    if initial_data in {'b'}: u0 = proj(Kstar(b))
    
    eta0, v0 = u0.copy(), u0.copy()
    w0 = nabla_x( u0, dim_x )
    gam0, xi0 = nabla_y( u0, dim_y ), nabla_y( w0, dim_y )
    
    alpha0, beta0 = 1, 1 *  np.max(u0.shape[dim_x+1:])
    beta1 = alpha1 * np.max(u0.shape[dim_x+1:])

    
    def L(X):
        return {'v': X['u'] + div_y(X['gam']) - X['eta'] ,
                'w': nabla_x(X['u'], dim_x) + div_y(X['xi']) }
    def Lstar(Y):
        return {'u': Y['v'] - div_x(Y['w']) ,
                'gam': - nabla_y(Y['v'], dim_y) ,
                'xi': - nabla_y(Y['w'], dim_y) ,
                'eta': - Y['v'] }
    def prox_F(X, tau):
        return {'u': np.real(X['u'] - tau*eps) ,
                'gam': prox_norm21(X['gam'], lam=alpha0*tau) ,
                'xi': prox_norm21(X['xi'], lam=alpha1*tau) ,
                'eta': proj(X['eta']) }
    def prox_Gstar(Y, sig):
        return {'v': proj_L_infty_ball(Y['v'], lam=beta0*sig),
                'w': proj_L_infty_ball(Y['w'], lam=beta1*sig) }

    X = {'u':u0, 'gam':gam0, 'xi':xi0, 'eta':eta0}
    Y = {'v':v0, 'w':w0}
    
    if method == 'CP':
        X, Y = CP(X, Y, tau, sig, prox_F, prox_Gstar, L, Lstar, iterations, printprogress=True)
        return X['u'], X['eta']
    else: print('Method argument not recognised')

def PR_TVPR(b, parameters=par_PR_TVPR, initial_data='zero', iterations=10000, method='CP', tau=tau, eps=eps):
    
    import numpy as np
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from prox_and_proj import prox_norm21, proj_L_infty_ball
    from algorithms_general import CP
    
    alpha0, alpha1, beta0, beta1 = parameters
    
    if b.ndim == 5: dim_x, dim_y, Lnorm_sq = 2, 3, 29
    if b.ndim == 4: dim_x, dim_y, Lnorm_sq = 2, 2, 25
    elif b.ndim == 3: dim_x, dim_y, Lnorm_sq = 1, 2, 21
    else: raise Exception('b.ndim is neither 4 nor 5')
    
    sig = 1 / (tau * Lnorm_sq) # second stepsize
    
    def L(X): # Note: I have verified that Lstar( . , parameters) is indeed the adjoint of L( . , parameters)
        return {'v': X['u'] + div_y(X['gam']) ,
                'w': nabla_x(X['u'], dim_x) + div_y(X['xi']) }
    def Lstar(Y):
        return {'u': Y['v'] - div_x(Y['w']) ,
                'gam': - nabla_y(Y['v'], dim=dim_y) ,
                'xi': - nabla_y(Y['w'], dim=dim_y) }
    # prox_norminf_lamfixed = lambda x : prox_norminf(x, lam=alpha1*tau)
    def prox_F(X, tau):
        return {'u': (X['u'] - tau*eps).clip(min=0) ,
                'gam': prox_norm21(X['gam'], lam=alpha0*tau) ,
                # 'xi': np.apply_along_axis(prox_norminf_lamfixed, 0, X['xi']) }
                'xi': prox_norm21(X['xi'], lam=alpha1*tau) }
    def prox_Gstar(Y, sig): # b is built in curve_experiment.py
        return {'v': proj_L_infty_ball(Y['v']-sig*b, lam=beta0),
                'w': proj_L_infty_ball(Y['w'], lam=beta1) }
    
    if initial_data in {'rand', 'random'}: u0 = np.random.rand(*b.shape)
    if initial_data in {0, 'zero', 'zeros'}: u0 = np.zeros(b.shape)
    if initial_data in {'b'}: u0 = b.copy()
    
    v0 = u0.copy()
    w0 = nabla_x( u0, dim_x )
    gam0, xi0 = nabla_y(u0, dim=dim_y), nabla_y(w0, dim=dim_y)

    X = {'u':u0, 'gam':gam0, 'xi':xi0}
    Y = {'v':v0, 'w':w0}
    
    if method == 'CP':
        X, Y = CP(X, Y, tau, sig, prox_F, prox_Gstar, L, Lstar, iterations)
        return X['u']
    else: print('Method argument not recognised')

'''In Inpainting_TVPR, b is a np.ma.masked array'''
def Inpainting_TVPR(b, parameters=par_Inpainting_TVPR, initial_data='zero', iterations=10000, method='CP', tau=tau, eps=eps):
    
    from differential_operators import nabla_x, div_x, nabla_y, div_y
    from prox_and_proj import proj_L_infty_ball, prox_norm21, prox_norm1
    from algorithms_general import CP
    
    beta0, beta1 = parameters
    
    if b.ndim == 4: dim_x, Lnorm_sq = 2, 16
    elif b.ndim == 3: dim_x, Lnorm_sq = 1, 12
    else: raise Exception('b.ndim is neither 4 nor 5')
    
    sig = 1 / (tau * Lnorm_sq) # second stepsize
    
    def L(X): 
        return {'w': nabla_x(X['u'], dim=dim_x) + div_y(X['xi']) }
    def Lstar(Y):
        return {'u': -div_x(Y['w']),
                'xi': -nabla_y(Y['w']) }
    
    if eps == 0: tempfunc = lambda x : x
    else: tempfunc = lambda x : prox_norm1(x, lam=tau*eps)
    mask = b.mask
    notmask = ~mask
    
    def prox_F(X, tau):
        temp_u = tempfunc(X['u'])
        temp_u[notmask] = b[notmask]
        return {'u': temp_u ,
                'xi': prox_norm21(X['xi'], lam=tau*alpha1) }
    def prox_Gstar(Y, sig, b=b): # b is built in curve_experiment.py
        return {'w': proj_L_infty_ball(Y['w'], lam=beta1) }
    
    if initial_data in {'random', 'rand'}: u0 = np.random.rand(*b.shape)
    if initial_data in {'zero', 'zeros'}: u0 = np.zeros(b.shape)
    if initial_data in {'b'}:
        u0 = b.data.copy()
        u0[b.mask] = 0
    
    w0 = nabla_x( u0, dim_x)
    xi0 = nabla_y(w0)

    X = {'u':u0, 'xi':xi0}
    Y = {'w':w0}
    
    if method == 'CP':
        X, Y = CP(X, Y, tau, sig, prox_F, prox_Gstar, L, Lstar, iterations)
        return X['u']
    else: print('Method argument not recognised')
    return
  
def L1_TVL1(b, parameters=par_L1_TVL1, initial_data='zero', iterations=10000, method='CP', tau=tau, eps=eps):
    
    from differential_operators import nabla_x, div_x
    from prox_and_proj import proj_L_infty_ball
    from algorithms_general import CP
    
    beta0, beta1 = parameters
    
    if b.ndim == 4: dim_x, Lnorm_sq = 2, 9
    elif b.ndim == 3: dim_x, Lnorm_sq = 1, 5
    else: raise Exception('b.ndim is neither 4 nor 5')
    
    sig = 1 / (tau * Lnorm_sq) # second stepsize
    
    def L(X): # Note: I have verified that Lstar( . , parameters) is indeed the adjoint of L( . , parameters)
        return {'v': X['u'] ,
                'w': nabla_x(X['u'], dim_x) }
    def Lstar(Y):
        return {'u': Y['v'] - div_x(Y['w']) }

    def prox_F(X, tau, eps=eps):
        return {'u': np.real(X['u']-tau*eps).clip(min=0) }
    def prox_Gstar(Y, sig, b=b): # b is built in curve_experiment.py
        return {'v': proj_L_infty_ball(Y['v']-sig*b, lam=beta0),
                'w': proj_L_infty_ball(Y['w'], lam=beta1) }
    
    if initial_data in {'random', 'rand'}: u0 = np.random.rand(*b.shape)
    if initial_data in {'zero', 'zeros'}: u0 = np.zeros(b.shape)
    if initial_data == {'b'}: u0 = b.copy()
    
    v0 = u0.copy()
    w0 = nabla_x( u0, dim_x)

    X = {'u':u0}
    Y = {'v':v0, 'w':w0}
    
    if method == 'CP':
        X, Y = CP(X, Y, tau, sig, prox_F, prox_Gstar, L, Lstar, iterations)
        return X['u']
    else: print('Method argument not recognised')
    return

def PR_TVL1(b, parameters=par_PR_TVL1, initial_data='zero', iterations=10000, method='CP', tau=tau, eps=eps):
    
    from differential_operators import nabla_x, nabla_y, div_x, div_y
    from prox_and_proj import prox_norm21, proj_L_infty_ball
    from algorithms_general import CP
    
    alpha0, beta0, beta1 = parameters
    
    if b.ndim == 4: dim_x, Lnorm_sq = 2, 17
    elif b.ndim == 3: dim_x, Lnorm_sq = 1, 13
    else: raise Exception('b.ndim is neither 4 nor 5')
    
    sig = 1 / (tau * Lnorm_sq) # second stepsize
    
    def L(X):
        return {'v': X['u'] + div_y(X['gam']) ,
                'w': nabla_x(X['u'], dim_x) }

    def Lstar(Y):
        return {'u': Y['v'] - div_x(Y['w']) ,
                'gam': -nabla_y(Y['v']) }

    def prox_F(X, tau):
        return {'u': (X['u']-tau*eps).clip(min=0),
                'gam': prox_norm21(X['gam'], lam=alpha0*tau) }

    def prox_Gstar(Y, sig):
        return {'v': proj_L_infty_ball(Y['v']-sig*b, lam=beta0),
                'w': proj_L_infty_ball(Y['w'], lam=beta1) }
    
    if initial_data in {'random'}: u0 = np.random.rand(*b.shape)
    if initial_data in {'zero', 'zeros'}: u0 = np.zeros(b.shape)
    if initial_data in {'b'}: u0 = b.copy()
    
    v0 = u0.copy()
    w0, gam0 = nabla_x( u0, dim_x), nabla_y(v0)

    X = {'u':u0, 'gam':gam0}
    Y = {'v':v0, 'w':w0}
    
    if method == 'CP':
        X, Y = CP(X, Y, tau, sig, prox_F, prox_Gstar, L, Lstar, iterations)
        return X['u']
    else: print('Method argument not recognised')
    return
