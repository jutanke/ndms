import numpy as np
import numba as nb


@nb.njit(nb.float64[:, :](nb.float64[:, :]), nogil=True)
def velocity(seq):
    """
    :param [n_frames x dim]
    """
    n_frames = len(seq)
    dim = seq.shape[1]
    V = np.empty((n_frames - 1, dim), dtype=np.float64)
    for t in range(n_frames - 1):
        a = seq[t]
        b = seq[t + 1]
        v = b - a
        V[t] = v
    return V