import numpy as np
import matplotlib.pyplot as plt
import ot


''' This function is also defined (differently) in objective_functions.py'''
def build_cost_matrix(metric, *coordinates_or_shape):
    '''
    arguments:
        metric: 'euclidean' returns the L2 distance (for W1), 'sqeuclidean' returns the L2^2 (for W2)
        coord_x, coord_y, coord_z: of lengths len_x, len_y, len_z
    size = len_x * len_y * len_z
    size[k] corresponds to coord_x[i_x], coord_y[i_y], coord_k[i_z] for suitable i_x, i_y, i_z
    Output:
        M with shape (size, size) such that M[k0, k1] is the (squared if metric='sqeuclidean') distance between
        ( coord_x[i0_x], coord_y[i0_y], coord_z[i0_z] )
        and
        ( coord_x[i1_x], coord_y[i1_y], coord_z[i1_z] )
    '''
    
    if all( isinstance(coord, int) for coord in coordinates_or_shape ):
        coordinates = [ np.arange(n)/(n-1) for n in coordinates_or_shape ]
    else: coordinates = coordinates_or_shape
    
    meshgrid_array = np.array(np.meshgrid( *coordinates, indexing='ij' ))

    domain_size = np.prod( [ len(coord) for coord in coordinates ] ) # it coincides with: np.prod( meshgrid_array.shape[1:] )

    meshgrid_flat = np.moveaxis( meshgrid_array.reshape( -1, domain_size ), 0, -1 )
    M = ot.dist( meshgrid_flat, metric=metric )
    
    return M

def W1(f, g, M=None): # if M is None, it defaults M to the L2 distance (hence the W1)
    assert f.shape == g.shape
    
    if M is None:
        M = build_cost_matrix('euclidean', *f.shape)   

    return ot.emd2(f.flatten(), g.flatten(), M)

# %% Test W1
f = np.zeros((10,11,12))
g = f.copy()
f[1,1,5] = 2
g[1,-1,5] = 0.5
g[-3,1,5] = 0.7
g[4,4,5] = 0.8
# print( W1(f,g) )

# %% Definition of W1_naif

def W1_naif_components(f, g, alpha1, alpha2, M=None):
    assert np.all(f>=0) and np.all(g>=0)
    f_mass, g_mass = np.sum(f), np.sum(g)
    f0 = f - f_mass/f.size
    g0 = g - g_mass/g.size
    f0_pos, f0_neg = np.maximum(f0,0), - np.minimum(f0,0)
    g0_pos, g0_neg = np.maximum(g0,0), - np.minimum(g0,0)

    ot_part = alpha1 * W1(f0_pos + g0_neg, f0_neg + g0_pos)
    mass_part = alpha2 * np.abs(g_mass - f_mass)
    return ot_part + mass_part, ot_part, mass_part

def W1_naif(f, g, alpha1, alpha2, M=None):
    return W1_naif_components(f, g, alpha1, alpha2, M)[0]

# %% Test W1_naif
f = np.zeros(20)
g = f.copy()
f[1] = 0.5
f[-2] = 0.5
g[-1] = 4
f=f
g=g+10
vmax = max( np.max(f), np.max(g) )
fig, ax = plt.subplots(2, 1)
x = np.arange(len(f))/(len(f)-1)
ax[0].stem(x, f, markerfmt='C0o', linefmt='C0-', basefmt=' ')
ax[0].set_xticks(x)
ax[0].set_ylim(0, 1.025*vmax)
ax[1].stem(x, g, markerfmt='C1o', linefmt='C1-', basefmt=' ')
ax[1].set_xticks(x)
ax[1].set_ylim(0,1.025*vmax)
plt.show()
print( W1_naif(f,g,1,1) )

# %% Experiment 1.1: single Dirac translation

def dirac_1d(n, i, mass=1.0):
    x = np.zeros(n, dtype=float)
    x[i] = mass
    return x

def cost_1d(n):
    I = np.arange(n)[:, None]
    return np.abs(I - I.T).astype(float)

def dirac_2d(shape, ij, m=1.0):
    x = np.zeros(shape, dtype=float)
    x[ij] = m
    return x

def cost_2d(shape, p=1):
    # p=1 Manhattan, p=2 Euclidean
    H, W = shape
    coords = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), axis=-1).reshape(-1, 2)
    diff = coords[:, None, :] - coords[None, :, :]
    if p == 1:
        return np.abs(diff).sum(axis=-1).astype(float)
    if p == 2:
        return np.sqrt((diff**2).sum(axis=-1)).astype(float)
    raise ValueError("p must be 1 or 2")


n = 101
i0 = 50
mass = 1.0
alpha1 = 1.0
alpha2 = 0.0  # equal masses here, so this term is zero anyway

ds = np.arange(0, 41)
vals = []
for d in ds:
    f = dirac_1d(n, i0, mass)
    g = dirac_1d(n, i0 + d, mass)
    vals.append(W1_naif(f, g, alpha1, alpha2) / alpha1)

vals = 100 * np.array(vals)
ref = m * ds  # standard balanced W1 between Diracs on a line

plt.figure()
plt.plot(ds, vals, marker="o", linewidth=1)
plt.plot(ds, ref, marker="x", linewidth=1)
plt.xlabel("displacement d")
plt.ylabel("W1_naif(f,g)/alpha1")
plt.title("Single Dirac translation on 1D grid")
plt.legend(["W1_naif/alpha1", "m*d (balanced W1 between Diracs)"])
plt.grid(True)
plt.show()

# %% Experiment 2.1: How much creation changes W1_naif

def dirac_1d(n, idx, mass=1.0):
    x = np.zeros(n, dtype=float)
    x[idx] = mass
    return x

def add_uniform_mass(arr, Delta):
    # Adds total mass Delta uniformly over the grid
    return arr + (Delta / arr.size)

def add_local_mass_1d(arr, Delta, idx):
    out = arr.copy()
    out[idx] += Delta
    return out


# -------------------------
# Experiment 2.1: uniform extra mass vs localized extra mass
# Domain is [0,1] with n points
# -------------------------
def experiment_uniform_vs_local_1d(
    n=101,
    alpha1=1.0,
    alpha2=10.0,
    Deltas=(0.1, 0.3, 0.6, 1.0),
    local_positions=(0, 10, 25, 50, 75, 90, 100),
    tol=1e-10,
):
    # baseline f: a simple sum of Diracs
    f = dirac_1d(n, idx=int(0.25 * (n - 1)), mass=1.0) + dirac_1d(n, idx=int(0.70 * (n - 1)), mass=0.5)

    # Precompute cost matrix once (important for speed)
    M = build_cost_matrix("euclidean", n)

    # Reference value (should be 0)
    Wff, Wff_ot, Wff_mass = W1_naif_components(f, f, alpha1, alpha2, M)

    print("Baseline check: W(f,f) =", Wff, " (ot:", Wff_ot, ", mass:", Wff_mass, ")")

    xs = np.linspace(0.0, 1.0, n)

    # Store results
    results = []  # list of dicts

    for Delta in Deltas:
        # (A) uniform extra mass
        g_uniform = add_uniform_mass(f, Delta)
        Wtot_u, Wot_u, Wm_u = W1_naif_components(f, g_uniform, alpha1, alpha2, M)

        extra_uniform = Wtot_u - Wff
        target = alpha2 * Delta

        # This should be exact up to numerical tolerance (OT part should stay the same)
        err = abs(extra_uniform - target)
        print(f"Delta={Delta:>5}: extra_uniform={extra_uniform:.12g}, alpha2*Delta={target:.12g}, err={err:.3g}")

        if err > tol:
            print("  WARNING: uniform-case deviation > tol. Consider loosening tol or checking numerical noise.")

        # (B) localized extra mass, same total Delta
        extras_local = []
        extras_local_ot = []
        extras_local_mass = []

        for j in local_positions:
            g_local = add_local_mass_1d(f, Delta, j)
            Wtot_l, Wot_l, Wm_l = W1_naif_components(f, g_local, alpha1, alpha2, M)

            extras_local.append(Wtot_l - Wff)
            extras_local_ot.append(Wot_l - Wff_ot)
            extras_local_mass.append(Wm_l - Wff_mass)  # should be alpha2*Delta, always

        results.append(
            dict(
                Delta=Delta,
                extra_uniform=extra_uniform,
                extra_local=np.array(extras_local),
                extra_local_ot=np.array(extras_local_ot),
                extra_local_mass=np.array(extras_local_mass),
            )
        )

    # -------------------------
    # Plots
    # -------------------------
    # Plot 1: extra cost vs location for localized creation, with uniform baseline line
    plt.figure()
    for r in results:
        Delta = r["Delta"]
        plt.plot(xs[list(local_positions)], r["extra_local"], marker="o", label=f"Δ={Delta}")
    for r in results:
        plt.axhline(alpha2 * r["Delta"], linestyle="--")  # reference: pure mass penalty
    plt.xlabel("x_j (location of added local mass)")
    plt.ylabel("W1_naif(f, f+Δδ_j) - W1_naif(f,f)")
    plt.title("Localized mass addition vs pure mass penalty α2·Δ (dashed)")
    plt.legend()
    plt.tight_layout()

    # Plot 2: decompose localized extra into OT-induced part + mass part
    plt.figure()
    for r in results:
        Delta = r["Delta"]
        plt.plot(xs[list(local_positions)], r["extra_local_ot"], marker="o", label=f"OT extra, Δ={Delta}")
    plt.xlabel("x_j")
    plt.ylabel("OT contribution change (alpha1*W1 part)")
    plt.title("How much localized mass creation changes the OT term")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    for r in results:
        Delta = r["Delta"]
        plt.plot(xs[list(local_positions)], r["extra_local_mass"], marker="o", label=f"mass extra, Δ={Delta}")
    plt.xlabel("x_j")
    plt.ylabel("Mass penalty change (alpha2*|Δmass|)")
    plt.title("Mass penalty term (should be flat at $α_2·Δ$)")
    plt.legend()
    plt.tight_layout()

    plt.show()

    return f, results


if __name__ == "__main__":
    f, results = experiment_uniform_vs_local_1d(
        n=101,
        alpha1=1.0,
        alpha2=10.0,
        Deltas=(0.1, 0.3, 0.6, 1.0),
        local_positions=(0, 10, 25, 50, 75, 90, 100),
        tol=1e-9,
    )
