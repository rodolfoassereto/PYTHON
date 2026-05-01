import numpy as np
import scipy.fft as spfft

# Add Libraries folder to sys.path
import sys
libraries_path = r"C:\Users\rodol\My Drive\PYHTON\Libraries"
if libraries_path not in sys.path:
    sys.path.insert(0, libraries_path)

def resolvent_with_undersampling_withmask(b, sigma, mask): # solves (id + σ * U.T @ U)x = b
    # mask is True on sampled entries
    return np.where(mask, b / (1.0 + sigma), b)


def laplacian_eigenvalues(shape, axes=None, h=1.0):
    """
    Eigenvalues of the multidimensional partial Laplacian associated with the
    1D operator having diagonal [-1, -2, ..., -2, -1] and off-diagonal 1,
    i.e. the one diagonalized by the DCT-II.

    Parameters
    ----------
    shape : tuple of int
        Shape of the multidimensional array u.
    axes : iterable of int, optional
        Axes along which the Laplacian acts.
        If None, uses all axes.
    h : float or sequence of float, optional
        Grid spacings. If scalar, the same spacing is used on every axis.
        If sequence, it must have length len(shape).

    Returns
    -------
    lam : ndarray
        Array of shape `shape`, containing the eigenvalues of the partial Laplacian.
        If axes = (0,1), then lam[i,j,k,...] = lam0[i] + lam1[j], where lam0 are the eigenvalues of Δ_0 (with Δ = Δ_0 + Δ_1)
    Note:
        eigenvalue
    """
    shape = tuple(shape)
    ndim = len(shape)

    if axes is None:
        axes = tuple(range(ndim))
    else:
        axes = tuple(ax % ndim for ax in axes)

    if np.isscalar(h):
        h = (float(h),) * ndim
    else:
        h = tuple(float(x) for x in h)
        if len(h) != ndim:
            raise ValueError("If h is a sequence, it must have length len(shape).")

    lam = np.zeros(shape, dtype=float)

    for ax in axes:
        n = shape[ax]
        if n <= 0:
            raise ValueError("All dimensions in shape must be positive.")

        k = np.arange(n)
        lam_1d = (2 * np.cos(np.pi * k / n) - 2) / (h[ax] ** 2)

        reshape = [1] * ndim
        reshape[ax] = n
        lam += lam_1d.reshape(reshape)

    return lam

def resolvent_with_laplacian(b, pointwise_division, axes, sigma=False):
    if sigma is False or isinstance(sigma, (int, float)):
        raise Exception('Check if sigma has been correctly used in pointwise_division and then set it to None')
    # solves (id - σ * ∆ )x = b where ∆ is the laplacian along the axes "axes"
    # pointwise_division is 1 - sigma * eigenvals, where eigenvals is the array containing the eigenvalues of ∆ (same shape as ∆), which can be obtained with my function "laplacian_eigenvalues"
    # the variable sigma is a reminder that the stepsize information must be encoded into the "pointwise_division" array
    return spfft.idctn( spfft.dctn(b, axes=axes) / pointwise_division, axes=axes ) # dctn defaults to type=2


def resolvent_with_II(b, II, JJ, shape_x, shape_y, pointwise_division, sigma_times_size_y=False):
    """
    solves (id - σ * IIstar ∆ II)x = b
    axes: tuple. It is the axes along which the laplacian acts
    """
    size_y = np.prod(shape_y)
    b1 = II(b, shape_x, shape_y) / size_y

    axes = tuple( [ i for i in range(len(shape_x)) ] )
    # note: here pointwise_division should be 1 - sig*size_y*eigenvalues
    z_P_small = resolvent_with_laplacian( b1, pointwise_division, axes, sigma=sigma_times_size_y )

    z_P = np.zeros( shape_x + shape_y )
    z_P[...] = z_P_small.reshape( shape_x + (1,)*len(shape_y) )

    return z_P + JJ(b, shape_x, shape_y)

def resolvent_with_JJ(b, PP, JJ, shape_x, shape_y, axes, pointwise_division, sigma=False):
    """
    solves (id - σ * JJ ∆_x JJ)x = b
    axes: tuple. It is the axes along which the laplacian acts
    """
    z_J = resolvent_with_laplacian( JJ(b, shape_x, shape_y), pointwise_division, axes, sigma=sigma )
    return z_J + PP(b, shape_x, shape_y)


