from skfem import *
from skfem import ElementVector
import numpy as np

# ---- Parameters ----
T = 1.0
n_refine = 2

# Create 1D grids for x, y, and time
pts_1d = np.linspace(0., 1., 2**n_refine + 1)
pts_t = np.linspace(0., T, 2**n_refine + 1)

# Create 3D mesh by tensor product
mesh = MeshTet.init_tensor(pts_1d, pts_1d, pts_t)

# ---- Finite Element Spaces ----
# P1 elements for scalar rho (density) and each component of momentum
el_scalar = ElementTetP1()

e_rho = el_scalar             # rho: scalar field in 3D

e_m = ElementVector  (el_scalar, dim=2) # it does "e_m = el_scalar * 2"; m: 2D vector field (m_x, m_y)

# Mixed basis: rho and m in the same product space
basis_rho = Basis(mesh, e_rho)
basis_m = Basis(mesh, e_m)

# Product basis for testing div(xi)=0
basis = basis_rho * basis_m

# ---- Weak Form of div(xi) = 0 ----
# We test against scalar test function phi:
# \int_\Omega rho * \partial_t phi + m_x * \partial_x phi + m_y * \partial_y phi dx dt = 0
# which we implement as inner( xi, grad(phi) )

from skfem.helpers import grad, dot

@BilinearForm
def continuity_form(rho, m, phi, w):
    return rho * grad(phi)[2] + dot(m, grad(phi)[:2])  # grad(phi) = (∂x, ∂y, ∂t), treat grad(phi)[:2] for m

# Assemble the weak form
A = continuity_form.assemble(basis)

# ---- Boundary Conditions ----
# You can apply BCs here if needed, e.g. zero initial/final density, or inflow conditions
# For now, we don't enforce anything to keep things general.

# ---- Solve or Project ----
# This system is underdetermined (just the constraint), and usually part of a bigger optimization
# For OT, you'll use it as a constraint in a saddle-point problem

print("Shape of constraint matrix A:", A.shape)

# You can inspect/test it here, or move on to define a minimization (e.g. kinetic energy functional)

# For instance, to test with a known (rho, m) satisfying div(xi)=0:
# Define test functions rho, m manually and compute residual r = A @ x
# Or include this as a constraint in an optimization loop
