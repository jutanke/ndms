import numpy as np


def velocity(seq):
    """
    :param [n_frames x dim]
    """
    n_frames = len(seq)
    dim = seq.shape[1]
    V = np.empty((n_frames - 1, dim), dtype=np.float32)
    for t in range(n_frames - 1):
        a = seq[t]
        b = seq[t + 1]
        v = b - a
        V[t] = v
    return V