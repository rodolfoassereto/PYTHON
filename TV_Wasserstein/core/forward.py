"""Forward operators (the data-fitting operator K of the variational problem).

  KK_factory / KKstar_factory : undersampled full complex FFT, selected by a
      boolean mask `mask_fft` over fftn frequencies (frequency layout, i.e.
      already ifftshifted).  The mask must be Hermitian-symmetric
      (mask(k) = mask(-k)); then K*K is a 0/1 diagonal in Fourier and the
      fidelity resolvent is an exact pointwise division.
"""
import numpy as np


def KK_factory(mask_fft):
    def KK(P):
        assert np.all(np.isreal(P))
        return np.fft.fftn(P, norm='ortho')[mask_fft]
    return KK


def KKstar_factory(mask_fft):
    def KKstar(E):
        full_array = np.zeros(mask_fft.shape, dtype=complex)
        full_array[mask_fft] = E
        return np.fft.ifftn(full_array, norm='ortho').real  # .real = true adjoint of R^n -> C^m
    return KKstar
