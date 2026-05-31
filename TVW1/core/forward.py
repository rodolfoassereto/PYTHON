"""Forward operators (the data-fitting operator K of the variational problem).

Two flavours, the single home for both:

  KK_factory / KKstar_factory : undersampled *real* FFT selected by a boolean
      mask `mask_rfft` over rfftn frequencies.  Used by the L2-TVW1 naif model.

  UF / UFstar : masked *full* FFT (numpy masked array).  Legacy forward operator
      used by the L2-TVPR family.

(Previously KK lived inside L2_TVW1_naif__model.py and UF inside build_toymodels.py;
experiments used to import the forward operator from a model file.)
"""
import numpy as np


# ---- Undersampled real FFT (L2-TVW1 naif model) ----
def KK_factory(mask_rfft):
    def KK(P):
        assert np.all(np.isreal(P))  # rfftn only if P is real
        return np.fft.rfftn(P, norm='ortho')[mask_rfft]
    return KK


def KKstar_factory(mask_rfft):
    def KKstar(E):
        full_array = np.zeros_like(mask_rfft, dtype=E.dtype)
        full_array[mask_rfft] = E
        return np.fft.irfftn(full_array, norm='ortho')
    return KKstar


# ---- Masked full FFT (legacy, L2-TVPR family) ----
def UF(x, mask=None):
    return np.ma.masked_array(np.fft.fftn(x, norm='ortho'), mask=mask)


def UFstar(y):
    return np.fft.ifftn(np.ma.filled(y, fill_value=0), norm='ortho')
