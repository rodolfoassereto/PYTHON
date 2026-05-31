"""
Python reimplementation of diffproptemp_double.m (2D crossing-fibre toy model)

- Build 2D propagator as mixture of two anisotropic Gaussians.
- Compute q-space via FFT.
- Apply different undersampling masks in q-space.
- Reconstruct propagator from masked q-space (zero-filled inverse FFT in place of strecon).
- Compute ODFs via Radon transform, remove isotropic part, and plot/save them.

Dependencies:
    numpy
    matplotlib
    scikit-image

Install missing deps with:
    pip install numpy matplotlib scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon


# -------------------------------------------------------------------------
# 1. Model: 2D crossing fibre propagator
# -------------------------------------------------------------------------

def build_crossing_propagator(
    xsize=33,
    displacement=0.00025,
    mixing_time=1.0,
):
    """
    Build a 2D propagator as a mixture of two anisotropic Gaussians.

    This mirrors the Matlab code:
        - first compartment: Dxx1, Dyy1
        - second compartment: Dxx2, Dyy2
        - mixture weights f1=f2=0.5
    """
    # Grid in real space (displacement coordinates)
    x = np.linspace(-displacement, displacement, xsize)
    y = np.linspace(-displacement, displacement, xsize)
    X, Y = np.meshgrid(x, y)

    # First fibre compartment
    Dxx1 = 5.30e-09  # m^2/s
    Dyy1 = 0.75e-09  # m^2/s
    f1 = 0.5

    Signal1 = (
        1.0
        / np.sqrt((4 * np.pi * mixing_time) ** 2 * (Dxx1 * Dyy1))
        * np.exp(
            -(
                (X**2) / (4 * Dxx1 * mixing_time)
                + (Y**2) / (4 * Dyy1 * mixing_time)
            )
        )
    )

    # Second fibre compartment (swapped anisotropy)
    Dxx2 = 0.30e-09  # m^2/s
    Dyy2 = 7.15e-09  # m^2/s
    f2 = 0.5

    Signal2 = (
        1.0
        / np.sqrt((4 * np.pi * mixing_time) ** 2 * (Dxx2 * Dyy2))
        * np.exp(
            -(
                (X**2) / (4 * Dxx2 * mixing_time)
                + (Y**2) / (4 * Dyy2 * mixing_time)
            )
        )
    )

    # Mixture
    Signal = f1 * Signal1 + f2 * Signal2

    return X, Y, Signal


# -------------------------------------------------------------------------
# 2. Forward model: propagator -> q-space
# -------------------------------------------------------------------------

def propagator_to_qspace(Signal):
    """
    q-space signal as 2D FFT of propagator.

    Matlab: qspace = fftshift(fft2(Signal));
    """
    qspace = np.fft.fftshift(np.fft.fft2(Signal))
    return qspace


# -------------------------------------------------------------------------
# 3. "strecon"-like reconstruction (zero-filled inverse FFT)
# -------------------------------------------------------------------------

def strecon_like(qspace, mask):
    """
    Simple stand-in for 'strecon' from the Matlab code.

    - 'qspace' is the fully sampled q-space (complex array, Nx x Ny)
    - 'mask'   is a binary sampling mask (same shape), where 1 = sampled

    We:
        1) apply the mask in q-space
        2) ifftshift to undo the fftshift
        3) inverse FFT to obtain an estimate of the propagator
    """
    assert qspace.shape == mask.shape
    kspace_masked = qspace * mask
    im = np.fft.ifft2(np.fft.ifftshift(kspace_masked))
    return im


# -------------------------------------------------------------------------
# 4. ODF via Radon transform
# -------------------------------------------------------------------------

def compute_odf_from_propagator(propagator, theta_deg=None):
    """
    Compute a 1D ODF from a 2D propagator via Radon transform.

    Steps:
        - take |propagator|
        - Radon transform over angles theta
        - pick the central radial row
        - subtract min (remove isotropic part), normalize to [0,1]

    This corresponds to the Matlab logic:
        [R,xp] = radon(abs(fftshift(im)), theta);
        ODF_full = R(25,:);
        ODF_full_aniso = (ODF_full - min(ODF_full)) / max(ODF_full - min(ODF_full));
    """
    if theta_deg is None:
        theta_deg = np.arange(0, 180)

    img = np.abs(propagator)
    sinogram = radon(img, theta=theta_deg, circle=False)  # shape: (n_projections, n_angles)

    # Choose a radius index near the center (instead of hard-coded 25)
    radius_idx = sinogram.shape[0] // 2
    odf = sinogram[radius_idx, :]

    # Remove isotropic part and normalize
    odf = odf - odf.min()
    max_val = odf.max()
    if max_val > 0:
        odf = odf / max_val

    return theta_deg, odf


def plot_odf(theta_deg, odf, title=None, fname=None):
    """
    Polar plot of ODF.
    """
    theta_rad = np.deg2rad(theta_deg)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta_rad, odf, lw=2)
    ax.set_rmax(1.05)
    if title is not None:
        ax.set_title(title, va="bottom")

    if fname is not None:
        fig.savefig(fname, dpi=300, bbox_inches="tight")

    return fig, ax


# -------------------------------------------------------------------------
# 5. Example undersampling masks
# -------------------------------------------------------------------------

def full_mask(shape):
    return np.ones(shape, dtype=np.float32)


def random_mask(shape, R, rng=None):
    """
    Simple random undersampling: each point is sampled independently
    with probability p = 1/R.

    R: nominal acceleration factor (e.g. R=2 -> ~50% samples).
    """
    if rng is None:
        rng = np.random.default_rng()
    nx, ny = shape
    p = 1.0 / float(R)
    mask = (rng.random((nx, ny)) < p).astype(np.float32)
    return mask


def gaussian_var_density_mask(shape, R, width=0.4, rng=None):
    """
    Crude Gaussian variable-density mask (cartesian grid).

    - Density peaked in the center of k-space.
    - Total number of samples ~ N / R (approximate).

    This is *not* exactly the same generator as in the Matlab code,
    but reproduces the idea of a Gaussian undersampling pattern.
    """
    if rng is None:
        rng = np.random.default_rng()
    nx, ny = shape
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    pdf = np.exp(-(X**2 + Y**2) / (2 * width**2))
    pdf /= pdf.max()

    # Threshold to reach desired sampling fraction ~1/R
    # We'll find a cutoff so that roughly (N / R) points are above it.
    N = nx * ny
    target = int(np.round(N / float(R)))
    pdf_flat = pdf.ravel()
    thresh = np.partition(pdf_flat, -target)[-target]  # value such that ~target elems >= thresh
    mask = (pdf >= thresh).astype(np.float32)

    return mask


def force_central_block(mask, radius=2):
    """
    Ensure a fully sampled central square (like gaussian_index(15:19,15:19)=1)

    radius=2 -> a block of size (2*radius+1) around center.
    """
    nx, ny = mask.shape
    cx, cy = nx // 2, ny // 2
    mask[cx - radius: cx + radius + 1, cy - radius: cy + radius + 1] = 1.0
    return mask


# -------------------------------------------------------------------------
# 6. Main driver
# -------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Build ground-truth propagator and q-space
    # ------------------------------------------------------------------
    xsize = ysize = 33
    displacement = 0.00025
    mixing_time = 1.0

    X, Y, Signal = build_crossing_propagator(
        xsize=xsize,
        displacement=displacement,
        mixing_time=mixing_time,
    )

    # q-space from FFT
    qspace = propagator_to_qspace(Signal)
    np.save("cross_fibre1_qspace.npy", qspace)

    # ------------------------------------------------------------------
    # Ground-truth ODF directly from Signal
    # ------------------------------------------------------------------
    theta_deg, odf_gt = compute_odf_from_propagator(Signal)
    plot_odf(theta_deg, odf_gt, title="Ground Truth ODF", fname="ODF_ground_truth.png")
    np.save("ODF_ground_truth.npy", odf_gt)

    # ------------------------------------------------------------------
    # "strecon" on fully sampled data (zero-filled inverse FFT)
    # ------------------------------------------------------------------
    mask_full = full_mask(qspace.shape)
    im_full = strecon_like(qspace, mask_full)
    np.save("im_full.npy", im_full)

    theta_deg, odf_full = compute_odf_from_propagator(np.fft.fftshift(im_full))
    plot_odf(theta_deg, odf_full, title="ODF from full sampling", fname="ODF_full_sampling.png")
    np.save("ODF_full_sampling.npy", odf_full)

    # ------------------------------------------------------------------
    # Example undersampling patterns
    # ------------------------------------------------------------------
    shape = qspace.shape
    rng = np.random.default_rng(seed=0)

    experiments = []

    # Gaussian variable-density masks with different R
    for R in [2, 3, 4, 6, 8]:
        mask = gaussian_var_density_mask(shape, R=R, width=0.4, rng=rng)
        mask = force_central_block(mask, radius=2)
        experiments.append(("gaussian_R{}".format(R), mask))

    # Random masks
    for R in [2, 3, 4]:
        mask = random_mask(shape, R=R, rng=rng)
        mask = force_central_block(mask, radius=2)
        experiments.append(("random_R{}".format(R), mask))

    # You could similarly implement a Poisson-disk-based mask generator
    # and append ("poisson_R...", mask) entries here.

    # ------------------------------------------------------------------
    # Run recon + ODF for each mask
    # ------------------------------------------------------------------
    for name, mask in experiments:
        im_rec = strecon_like(qspace, mask)
        np.save(f"im_{name}.npy", im_rec)

        # ODF from reconstructed propagator
        theta_deg, odf = compute_odf_from_propagator(np.fft.fftshift(im_rec))
        plot_odf(theta_deg, odf, title=f"ODF ({name})", fname=f"ODF_{name}.png")
        np.save(f"ODF_{name}.npy", odf)

        # Optional: visualize the sampling pattern and reconstructed propagator
        # Sampling pattern in q-space
        plt.figure()
        plt.imshow(mask, cmap="gray")
        plt.title(f"Sampling mask: {name}")
        plt.colorbar()
        plt.savefig(f"mask_{name}.png", dpi=300, bbox_inches="tight")

        # Reconstructed propagator magnitude
        plt.figure()
        plt.imshow(np.abs(np.fft.fftshift(im_rec)), extent=[
            X.min(), X.max(), Y.min(), Y.max()
        ])
        plt.xlabel("displacement [m]")
        plt.ylabel("displacement [m]")
        plt.title(f"Reconstructed propagator: {name}")
        plt.colorbar()
        plt.savefig(f"im_rec_{name}.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
