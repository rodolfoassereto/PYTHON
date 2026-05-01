from numpy import *
from matplotlib.pyplot import *
from scipy.sparse.linalg import eigs
# import ipdb as pdb
import building_data as bd

def grad(u):
    shp = [2] + list(u.shape)
    grad_u = zeros(shp)
    grad_u[0,:-1,:] = u[1:,:] - u[:-1,:]
    grad_u[1,:,:-1] = u[:,1:] - u[:,:-1]
    return grad_u

def grad_ad(v):
    shp = v.shape[1:]
    grad_ad_v = zeros(shp)
    grad_ad_v[1:,:] = v[0,:-1,:]
    grad_ad_v[:-1,:] -= v[0,:-1,:]
    grad_ad_v[:,1:] += v[1,:,:-1]
    grad_ad_v[:,:-1] -= v[1,:,:-1]
    return grad_ad_v

def proj(v, alpha):
    v_abs = hypot(v[0,:,:], v[1,:,:])
    v_fac = maximum(1, v_abs/alpha)
    return v/v_fac

def downsample0(u):
    u = pad(u, list(zip((0,0), array(u.shape) & 1)), 'edge')
    v = 0.25*(u[::2,::2] + u[1::2,::2] + u[::2,1::2] + u[1::2,1::2])
    return v

def downsample0_ad(v, shp):
    u = zeros(2*array(v.shape))
    u[::2,::2] = v
    u[1::2,::2] = v
    u[::2,1::2] = v
    u[1::2,1::2] = v
    if shp[0] & 1:
        u = u[:-1,:]
        u[-1,:] *= 2
    if shp[1] & 1:
        u = u[:,:-1]
        u[:,-1] *= 2
    return 0.25*u

def downsample1(u):
    u = pad(u, list(zip((1,1), (array(u.shape) & 1) + 1)), 'edge')
    v1 = u[:-2:2,:-2:2] + u[2::2,:-2:2] + u[:-2:2,2::2] + u[2::2,2::2]
    v2 = u[:-2:2,1:-1:2] + u[2::2,1:-1:2] + u[1:-1:2,:-2:2] + u[1:-1:2,2::2]
    v = 1/16*(4*u[1:-1:2,1:-1:2] + 2*v2 + v1)
    return v

def downsample1_ad(v, shp):
    u = zeros(2*array(v.shape) + 2)
    u[:-2:2,:-2:2] = v
    u[2::2,:-2:2] += v
    u[:-2:2,2::2] += v
    u[2::2,2::2] += v
    v *= 2
    u[:-2:2,1:-1:2] += v
    u[2::2,1:-1:2] += v
    u[1:-1:2,:-2:2] += v
    u[1:-1:2,2::2] += v
    u[1:-1:2,1:-1:2] += 2*v

    u[1,:] += u[0,:]
    u[-2,:] += u[-1,:]
    u = u[1:-1,:]

    u[:,1] += u[:,0]
    u[:,-2] += u[:,-1]
    u = u[:,1:-1]

    if shp[0] & 1:
        u[-2,:] += u[-1,:]
        u = u[:-1,:]
    if shp[1] & 1:
        u[:,-2] += u[:,-1]
        u = u[:,:-1]
    
    return 1/16*u


def proj_constraints(u):
    length = len(u)
    weights = [None]*(length-1)
    v = [None]*(length-1)
    for i in range(length-1):
        Sstar = downsample0_ad(ones(u[i+1].shape), u[i].shape)
        if i > 0:
            Sstar *= (1 - 1/weights[i-1])
        weights[i] = 1 + downsample0(Sstar)
        v[i] = downsample0(u[i]) - u[i+1]
        if i > 0:
            v[i] += downsample0(v[i-1])
        v[i] /= weights[i]

    for i in range(length-3,-1,-1):
        v[i] += downsample0_ad(v[i+1], v[i].shape)/weights[i]

    u_ = [None]*length
    for i in range(length):
        u_[i] = u[i].copy()
        if i > 0:
            u_[i] += v[i-1]
        if i < length-1:
            u_[i] -= downsample0_ad(v[i], u[i].shape)

    return u_

def proj_constraints_(u):
    length = len(u)
    weights = [None]*(length-1)
    v = [None]*(length-1)
    for i in range(length-1):
        #Sstar = downsample0_ad(ones(u[i+1].shape), u[i].shape)
        weights[i] = 1 + 0.25
        if i > 0:
            #Sstar *= (1 - 1/weights[i-1])
            weights[i] = 1 + 0.25*(1 - 1/weights[i-1])
        #weights[i] = 1 + downsample0(Sstar)
        v[i] = downsample0(u[i]) - u[i+1]
        if i > 0:
            v[i] += downsample0(v[i-1])
        v[i] /= weights[i]

    for i in range(length-3,-1,-1):
        v[i] += downsample0_ad(v[i+1], v[i].shape)/weights[i]

    u_ = [None]*length
    for i in range(length):
        u_[i] = u[i].copy()
        if i > 0:
            u_[i] += v[i-1]
        if i < length-1:
            u_[i] -= downsample0_ad(v[i], u[i].shape)

    return u_

def prox_val_l2(u, step, params):
    (f, beta) = params
    g = sum((u-f)**2)*beta/2
    prox_val = g/(1 + beta*step)**2
    return prox_val

def prox_deriv_l2(u, step, params):
     (f, beta) = params
     g = sum((u-f)**2)*beta/2
     deriv = -2*beta*g/(1 + beta*step)**3
     #return deriv
     return (deriv, deriv) if step > 0 else (-Inf, deriv)

def prox_op_l2(u, step, params):
    (f, beta) = params
    prox = (u + beta*step*f)/(1 + beta*step)
    return prox

def prox_max_step_l2(u, params):
    return Inf

def prox_val_l1(p, step, alpha):
    return alpha*sum(maximum(0.0, abs(p) - alpha*step))

def prox_deriv_l1(p, step, alpha):
    m0 = -alpha**2*maximum(1.0, sum(abs(p) >= step*alpha))
    m1 = -alpha**2*sum(abs(p) > step*alpha)
    return (m0, m1) if step > 0 else (-Inf, m1)

def prox_op_l1(p, step, alpha):
    p_ = sign(p)*maximum(0.0, abs(p) - alpha*step)
    return p_

def prox_max_step_l1(p, alpha):
    return abs(p).max()/alpha

def prox_max(u, step, prox_op, prox_val, prox_deriv, prox_max_step, params):
    if step == 0:
        return [v.copy() for v in u]
    
    length = len(u)
    cur_step = zeros((length,))
    max_step = array([prox_max_step(u[i], params[i]) for i in range(length)])
    if step >= sum(max_step):
        return [prox_op(u[i], max_step[i], params[i]) for i in range(length)]
    step_val = array([prox_val(u[i], step, params[i]) for i in range(length)])
    step_left = step
    for i in step_val.argsort()[::-1]:
        cur_step[i] = minimum(step_left, max_step[i])
        step_left -= cur_step[i]
    cur_c = step_val[cur_step > 0].min()
    while True:
        step_val = array([prox_val(u[i], cur_step[i], params[i]) for i in range(length)])
        #print(f"cur_c={cur_c}")
        print(f"cur_step={cur_step}")
        print(f"step_val={step_val}")
        cur_deriv = [prox_deriv(u[i], cur_step[i], params[i]) for i in range(length)]
        #print(f"cur_deriv={cur_deriv}")
        mask = step_val <= cur_c
        while True:
            deriv_select = [d[0] if mask[i] else d[1] for (i,d) in enumerate(cur_deriv)]
            cur_reci = 1.0/array(deriv_select)
            cur_c = sum(cur_reci*step_val)/sum(cur_reci)
            old_mask = mask
            mask = step_val <= cur_c
            if (mask == old_mask).all():
                break
        update = cur_reci*(cur_c - step_val)
        error=sum(abs(update))/(length*step)
        print(f"error={error}")
        if error < 1e-10: # stopping criterion
            break
        new_step = cur_step + update
        negative = new_step < 0
        step_length = ones_like(cur_step)
        step_length[negative] = cur_step[negative]/(cur_step[negative] - new_step[negative])
        step_length_min = step_length.min()
        cur_step = step_length*new_step + (1-step_length)*cur_step
        if step_length_min < 1.0:
            cur_step[step_length == step_length_min] = 0

    v = [prox_op(u[i], cur_step[i], params[i]) for i in range(length)]
    return v
    

def proj_epi_l2(u, t, f):
    g = sum((u - f)**2)/2
    if g <= t:
        return (u.copy(), t)
    r = roots([1, 2*(1-t), (1-t)**2, -g])
    t_ = real(r).max()
    u_ = (u + (t_-t)*f)/(1 + (t_-t))
    if False:
        check = sum((u_ - f)**2)/2
        #print(f"r={r}, t={t}, t_={t_}, check={check}")
        print(f"l2_error={abs(t_-check)}")
#        if abs(t_-check) > 1e-6:
#            raise Exception("Ouch!")
    return (u_, t_)


def proj_epi_l1(v, tau, alpha):
    h = alpha*sum(abs(v))
    if h <= tau:
        return (v.copy(), tau)
    c = abs(v) + alpha*tau
    tau_ = tau
    h_ = h
    m_old = Inf
    while True:
        m_ = sum(c > alpha*tau_)
        if m_ == 0:
            break
        tau_ = (h_ + m_*tau_*alpha**2)/(1 + m_*alpha**2)
        h_ = alpha*sum(maximum(c - alpha*tau_, 0))
        if m_ >= m_old:
            break
        m_old = m_
    v_ = sign(v)*maximum(c - alpha*tau_, 0)
    if False:
        check = alpha*sum(abs(v_))
        #print(f"tau={tau}, tau_={tau_}, check={check}")
        print(f"l1_error={abs(tau_-check)}")
        #if abs(tau_-check) > 1e-6:
        #    raise Exception("Ouch!")
    return (v_, tau_)


def print_iterate(u, v, p, f, alpha, beta):
    length = len(u)
    g = empty((length,))
    h = empty((length,))
    vnorm = empty((length,))
    pnorm = empty((length,))
    for i in range(length):
        g[i] = 0.5*sum((u[i] - f[i])**2)*beta[i]
        h[i] = alpha[i]*sum(abs(grad(u[i])))
        vnorm[i] = sqrt(sum(v[i]**2))
        pnorm[i] = sqrt(sum(p[i]**2))
    values.append(g[0]+h[0])

    proj = empty((length-1,))
    for i in range(length-1):
        proj[i] = sqrt(sum((downsample0(u[i]) - u[i+1])**2))

    print(f"functional value={g+h}")
    print(f"g={g}, h={h}")
#    print(f"projection error={proj}")
    print(f"v_norm={vnorm}")
    print(f"p_norm={pnorm}")
#    print(f"alpha0={alpha0}, alpha={alpha}")

def tv_multigrid_prox_step(u, v, p, f, alpha, beta, step):
    length = len(u)
    (sigma_step, tau_step) = step

    u_ = proj_constraints(u)

    v_ = [v[i]/tau_step for i in range(length)]
    params_l2 = [(f[i], beta[i]) for i in range(length)]
    v_prox = prox_max(v_, 1/tau_step, prox_op_l2, prox_val_l2, prox_deriv_l2, prox_max_step_l2, params_l2)
    v_ = [v[i] - tau_step*v_prox[i] for i in range(length)]
    
    p_ = [p[i]/tau_step for i in range(length)]
    p_prox = prox_max(p_, 1/tau_step, prox_op_l1, prox_val_l1, prox_deriv_l1, prox_max_step_l1, alpha)
    p_ = [p[i] - tau_step*p_prox[i] for i in range(length)]
    
    return (u_, v_, p_)


def tv_multigrid_step(u, v, p, d_u, f, alpha, beta, step):
    length = len(u)
    (sigma_step, tau_step) = step

    #pdb.set_trace()
    (u_, v_, p_) = tv_multigrid_prox_step(u, v, p, f, alpha, beta, step)
    print_iterate(u_, v_, p_, f, alpha, beta)
    
    b_u = [None]*length
    for i in range(length):
        b_u[i] = (2*u_[i] - u[i]) - sigma_step*(2*v_[i] - v[i] + grad_ad(2*p_[i] - p[i]))

    lambda_step0 = 1 + sigma_step*tau_step*9
    d_u_ = [None]*length
    for i in range(length):
        d_u_[i] = d_u[i] + (b_u[i] - (1 + sigma_step*tau_step)*d_u[i] \
                            - sigma_step*tau_step*grad_ad(grad(d_u[i])))/lambda_step0
    
    u_new = [None]*length
    v_new = [None]*length
    p_new = [None]*length
    for i in range(length):
        u_new[i] = u[i] - u_[i] + d_u_[i]
        v_new[i] = v_[i] + tau_step*d_u_[i]
        p_new[i] = p_[i] + tau_step*grad(d_u_[i])
    d_new = d_u_

    return (u_new, v_new, p_new, d_new, u_)


def tv_primal_dual_step(u, p, f, alpha, step):
    (sigma, tau) = step
    u_new = (u - sigma*(grad_ad(p) - f))/(1 + sigma)
    u_bar = 2*u_new - u
    p_new = proj(p + tau*grad(u_bar), alpha)
    return (u_new, p_new)


def primal_dual_test():
    u_true = imread('lena.png')
    f = u_true + random.randn(*u_true.shape)*0.1
    iter = 1000
    alpha = 0.1
    step = (1/sqrt(8), 1/sqrt(8))

    u = zeros(f.shape)
    p = zeros([2] + list(f.shape))
    for i in range(iter):
        (u, p) = tv_primal_dual_step(u, p, f, alpha, step)

    return (u, p)


def prepare_data(f0, alpha0, max_depth=0):
    u = []
    v = []
    p = []
    f = []
    alpha = []
    beta = []
    t = []
    tau = []
    d_u = []
    
    f_cur = f0
    alpha_cur = alpha0
    beta_cur = 1.0
    depth = 1
    while True:
        f.append(f_cur)
        u.append(f_cur.copy())
        v.append(zeros_like(f_cur))
        p.append(zeros([2] + list(f_cur.shape)))
        d_u.append(f_cur.copy())
        
        alpha.append(alpha_cur)
        beta.append(beta_cur)

        if max(f_cur.shape) == 1 or depth == max_depth:
            break

        f_cur = downsample0(f_cur)
        alpha_cur *= 2
        beta_cur *= 4

        depth += 1

    d = d_u

    step = (2, 1/2)

    alpha = array(alpha)
    beta = array(beta)

    return (u, v, p, d, f, alpha, beta, step)



def tv_initial_guess(f, alpha, beta, step):
    length = len(f)
    (sigma_step, tau_step) = step
    
    u = [None]*length
    v = [None]*length
    p = [None]*length
    d_u = [None]*length
    for i in range(length):
        u[i] = f[i].copy()
        v[i] = zeros_like(u[i]) # u[i].copy()
        v[i] = u[i].copy()
        p[i] = grad(u[i])
        pabs = hypot(p[i][0,...], p[i][1,...])
        pabs[pabs == 0] = 1
        p[i] *= alpha[i]/pabs
        #p[i] = zeros_like(grad(u[i]))
        d_u[i] = u[i].copy()
    d = d_u

    #print_iterate(u, v, p, f, alpha, beta)
    #return (u, v, p, d)

    u_proj = proj_constraints(u)
    u_ = [u_proj[i] - u[i] for i in range(length)]
    v_ = [beta[i]*tau_step*(v[i] - f[i]) for i in range(length)]
    p_ = [zeros_like(p[i]) for i in range(length)]

    return (u_, v_, p_, d)



def visualize(u, plots):
    length = len(u)
    if plots is None:
        clf()
        rows = int(floor(sqrt(length)))
        cols = int(ceil(length/rows))
        plots = []
        for i in range(length):
            subplot(rows, cols, i+1)
            im = imshow(u[i], cmap=cm.gray, vmin=0, vmax=1)
            plots.append(im)
    else:
        for i in range(length):
            plots[i].set_data(u[i])

    return plots



#u_true = imread('taj_mahal.png')[:,:,0] #[::8,::8]
# u_true = imread('abstract_pattern.jpg')
u_true = bd.import_image('lena.png', 9)
random.seed(42)
f0 = u_true + random.randn(*u_true.shape)*0.4
alpha0 = 0.25
iter = 100

imshow(zeros((1,1)))
show()

ion()
figure(0)
plots = visualize([f0], None)

depths = [0]
VAL = []
for (i, depth) in enumerate(depths):
    (u, v, p, d, f, alpha, beta, step) = prepare_data(f0, alpha0, max_depth=depth)
    (u, v, p, d) = tv_initial_guess(f, alpha, beta, step)

    figure(i+1)
    plots=None
    values = []
    for i in range(iter):
        print(f"iteration {i}")
        (u, v, p, d, u_) = tv_multigrid_step(u, v, p, d, f, alpha, beta, step)

        if i == iter-1: # i % 2 == 0 or
            imshow(u_[0], cmap='gray', vmin=0, vmax=1)
            title(f"iteration={i}, depth={depth}")
            show()
            # plots = visualize(u_[0:1], plots)
            # draw()
            # gcf().canvas.flush_events()
            # time.sleep(1e-8)
    
    VAL = VAL + [values]
    figure(100)
    plot(array(values))
    title(f"depth={depth}")
    show()
