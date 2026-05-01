# %% Definizioni funzioni di prova per algoritmi

import numpy as np

def hilbert_matrix(n, dim=2): # creates a dim_dimensional Hilbert matrix (ill-conditioned)
    grids = np.ogrid[tuple(slice(0, n) for _ in range(dim))] # Create a grid of indices
    # Sum the indices element-wise and add 1 to avoid division by zero
    total = sum(grids) + 1
    return 1.0 / total


def prova_CP(A, C, b, iterations=10000): # min_{Cx >= 0} = 1/2 ||Ax - b||^2
    import time
    
    d = A.shape[-1]
    if A.shape[0] != len(b): print('b dim does not coincide with the output of A')
    
    t1 = time.time()
    import cvxpy as cp
    x = cp.Variable(d)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [C @ x >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    print(problem.status)
    x_cvxpy = x.value
    t2 = time.time()
    print('cvxpy time: ', t2 - t1, 'sec')
    
    I = np.eye(d)
    
    def prox_f(x, tau):
        return { key: np.linalg.solve( I + tau * A.T @ A , x[key] + tau * A.T @ b ) for key in x }
    
    def prox_gstar(u, sigma):
        return { key: u[key].clip(max=0) for key in u }
    
    # Identity operator
    def L(x):
        return {'u': C @ x['x']}
    
    def Lstar(u):
        return {'x': C.T @ u['u']}
    
    L_norm = max(np.linalg.svd(C, compute_uv=False))
    # L_norm_sq = np.linalg.norm(C)**2
    param_factor = 2000 # for best convergence tau is 10^3 times sigma!
    tau = param_factor / L_norm
    sigma = 1 / ( param_factor * L_norm)
    
    # Initialize variables
    x0 = {'x': np.zeros(d)}  # Starting at zero
    u0 = L(x0)  # Dual variable initialized to zero

    # Run Chambolle-Pock Algorithm
    t1 = time.time()
    from algorithms_general import CP
    x_opt, u_opt = CP(x0, u0, tau, sigma, prox_f, prox_gstar, L, Lstar, iterations, printprogress=False)
    x_opt = x_opt['x']
    t2 = time.time()
    print('CP time: ', t2 - t1, 'sec')
    
    return x_opt, x_cvxpy


def prova_pDR_Richardson(A, C, b, iterations=10000): # min_{Cx >= 0} = 1/2 ||Ax - b||^2; confronto con soluzione di scipy
    
    import cvxpy as cp
    import time
    
    n = A.shape[1]
    
    t1 = time.time()
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [C @ x >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    x_cvxpy = x.value
    t2 = time.time()
    print('cvxpy time: ', np.round( t2 - t1, 4), 'sec')
    
    I = np.eye(n)
    
    def prox_f(x, tau):
        return { 'x': np.linalg.solve( I + tau * A.T @ A , x['x'] + tau * A.T @ b ) }
    
    def prox_g(y, sigma):
        return { 'y': y['y'].clip(max=0) }
    
    def K(x):
        return { 'y': C @ x['x'] }
    def Kstar(y):
        return { 'x': C.T @ y['y'] }
    
    # Initialize variables
    x0 = {'x': np.zeros(n)}  # Starting at zero

    # Run Chambolle-Pock Algorithm
    from algorithms_general import pDR_Richardson
    
    K_norm_sq = max(np.linalg.svd(C, compute_uv=False))**2
    # K_norm_sq = np.sum( C**2 )
    
    sig=1
    print('sig = ', sig)
    
    t1 = time.time()
    x_opt, y_opt = pDR_Richardson(x0, sig, prox_f, prox_g, K, Kstar, K_norm_sq, iterations=iterations)
    x_opt = x_opt['x']
    t2 = time.time()
    print('pDR time: ', np.round(t2 - t1, 4), 'sec')
    
    return x_opt, x_cvxpy

# %% Prova per graph_DR

from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

from algorithms_general import graph_DR

# Minimize over x ∈ R^M:   0.5 * ||A x - b||^2  +   λ * sum_i Huber_δ(x_i)   +   I_{[l, u]^M}(x)
#
# where:
#   - A ∈ R^{n_obs × M}, b ∈ R^{n_obs}
#   - Huber_δ(t) = 0.5 t^2          if |t| ≤ δ
#                  δ(|t| - 0.5 δ)   otherwise
#   - I_{[l,u]^M}(x) = 0 if x ∈ [l,u]^M, +∞ otherwise
#
# i.e., a box-constrained least-squares problem with separable Huber regularization.

# -----------------------
# Prox operators
# -----------------------

from prox_and_proj import make_prox_quadratic, make_prox_huber, make_prox_box

# -----------------------
# Objective + gradient for SciPy
# -----------------------
def huber_value(x, delta):
    ax = np.abs(x)
    out = np.empty_like(x)
    mask = ax <= delta
    out[mask] = 0.5 * x[mask]**2
    out[~mask] = delta * (ax[~mask] - 0.5 * delta)
    return out

def huber_grad(x, delta):
    ax = np.abs(x)
    g = np.empty_like(x)
    mask = ax <= delta
    g[mask] = x[mask]
    g[~mask] = delta * np.sign(x[~mask])
    return g

# -----------------------
# Problem instance
# -----------------------
rng = np.random.default_rng(0)
M = 50
n_obs = 120

A = rng.standard_normal((n_obs, M)) / np.sqrt(n_obs)
x_true = rng.standard_normal(M)
b = A @ x_true + 0.05 * rng.standard_normal(n_obs)

lmbda = 0.5
delta = 0.2
l_box, u_box = -1.0, 1.0

# three proximable functions
prox1 = make_prox_quadratic(A, b)
prox2 = make_prox_huber(lmbda, delta)
prox3 = make_prox_box(l_box, u_box)

# tree: chain 0->1->2 (topological)
parent_node = [[], [0], [1]]

# incidence matrix Z for chain
Z = np.array([[ 1.0,  0.0],
              [-1.0,  1.0],
              [ 0.0, -1.0]])

d = np.diagonal(Z @ Z.T)  # [1,2,1]

w0 = [{"x": np.zeros(M)} for _ in range(Z.shape[1])]

# -----------------------
# Run your method
# -----------------------
sigma = 1.0
theta = 1.0
iters_graph_dr = 500

x_nodes = graph_DR(sigma, Z, parent_node, d, w0, iters_graph_dr, [prox1, prox2, prox3], theta=theta)
xs = np.stack([xi["x"] for xi in x_nodes], axis=0)
x_graph = xs.mean(axis=0)   # take mean for stability

# -----------------------
# Run SciPy L-BFGS-B on the same problem
# -----------------------
def objective_and_grad(x):
    r = A @ x - b
    f1 = 0.5 * np.dot(r, r)
    f2 = lmbda * np.sum(huber_value(x, delta))
    f = f1 + f2
    g = (A.T @ r) + lmbda * huber_grad(x, delta)
    return f, g

bounds = [(l_box, u_box)] * M
x0 = np.zeros(M)

res = minimize(lambda x: objective_and_grad(x)[0],
               x0,
               jac=lambda x: objective_and_grad(x)[1],
               method="L-BFGS-B",
               bounds=bounds,
               options={"maxiter": 2000})

x_scipy = res.x

# -----------------------
# Compare
# -----------------------
def full_objective(x):
    r = A @ x - b
    return 0.5 * np.dot(r, r) + lmbda * np.sum(huber_value(x, delta))

print("SciPy success:", res.success)
print("SciPy nit:", res.nit)
print("Objective SciPy:", full_objective(x_scipy))
print("Objective graph_DR:", full_objective(x_graph))
print("||x_graph - x_scipy||:", np.linalg.norm(x_graph - x_scipy))
print("Consensus max ||x_i - mean||:", np.max(np.linalg.norm(xs - x_graph[None, :], axis=1)))


# %%

m = 8
A = hilbert_matrix(m)
# A = np.random.rand(m,m)
np.random.seed(0)
C = hilbert_matrix(m) + np.random.rand(m,m)/m
b = np.random.rand(m)

iterations = 10000
# x_opt_CP, x_cvxpy = prova_CP(A, C, b, iterations=iterations)
# print('Current method: >>> CP <<<')
x_opt_DR, x_cvxpy = prova_pDR_Richardson(A, C, b, iterations=iterations)
print('Current method: >>> pDR <<<')

x_opt = x_opt_DR.copy()
from default import printarr
precision = 3
print('_______________________________')
print('x_cvxpy:')
printarr(x_cvxpy, decimals=precision)
print('x_opt:')
printarr(x_opt, decimals=precision)
print('C @ x_cvxpy =', np.round(C @ x_cvxpy, precision+2) )
# print('C @ x_cvxpy >=0?', C @ x_cvxpy >= 0 )
print('C @ x_opt = ', np.round(C @ x_opt, precision+2) )
# print('C @ x_opt >=0? ', C @ x_opt >= 0)
print('||A @ x_cvxpy - b||^2 = ', np.round( np.linalg.norm(A @ x_cvxpy - b)**2, precision+2 ) )
print('||A @ x_opt - b||^2 = ', np.round( np.linalg.norm(A @ x_opt - b)**2, precision+2 ) )
# printarr(C @ x_opt_DR)np.linalg.norm(A @ x_opt_CP - b))
print('||x_opt - x_cvxpy|| = ', np.round(np.linalg.norm(x_opt - x_cvxpy), precision ) )
# printarr(C @ x_opt_DR)
# print(np.linalg.norm(A @ x_opt_DR - b))


# %% Prova DR

def prova_DR(A, b): # min_{x >= 0} = 1/2 ||Ax - b||^2; confronto con soluzione di scipy

    from scipy.optimize import nnls    
    x_scipy, residual = nnls(A, b)
    
    I = np.eye(A.shape[1])
    
    def prox_f(x, tau):
        return { 'x': np.linalg.inv( I + tau * A.T @ A ) @ ( x['x'] + tau * A.T @ b ) }
    
    # Proximal operator of g*(u) = 0 (identity function)
    def prox_g(s, sigma):
        return {'x': s['x'].clip(min=0) }
    
    # Initialize variables
    s0 = {'x': np.zeros(A.shape[1])}  # Starting at zero

    # Run Chambolle-Pock Algorithm
    from algorithms_general import DR
    
    x_opt, s_opt = DR(s0, prox_f, prox_g, iterations=10000 )
    
    return x_opt, x_scipy
