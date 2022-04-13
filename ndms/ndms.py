import numpy as np
import numba as nb
import numpy.linalg as la
import math as m
from ndms.derivative import velocity


@nb.njit(nb.float32(nb.float32[:, :], nb.float32[:, :], nb.int64), nogil=True)
def ndms(true_seq, query_seq, kernel_size):
    """
    :param true_seq: [kernel_size x n_useful_joints*3]
    :param query_seq: [kernel_size x n_useful_joints*3]
    """
    eps = 0.0000001
    true_v_ = np.ascontiguousarray(velocity(true_seq))
    query_v_ = np.ascontiguousarray(velocity(query_seq))
    dim = true_v_.shape[1]
    n_features = dim // 3
    true_v = true_v_.reshape((kernel_size - 1, n_features, 3))
    query_v = query_v_.reshape((kernel_size - 1, n_features, 3))
    total_score = 0.0
    for t in range(kernel_size - 1):
        for jid in range(n_features):
            a = np.expand_dims(true_v[t, jid], axis=0)
            b_T = np.expand_dims(query_v[t, jid], axis=1)
            norm_a = max(la.norm(a), eps)
            norm_b = max(la.norm(b_T), eps)
            cos_sim = (a @ b_T) / (norm_a * norm_b)
            disp = min(norm_a, norm_b) / max(norm_a, norm_b)
            score = ((cos_sim + 1) * disp) / 2.0
            total_score += score[0, 0] / n_features
    return total_score / (kernel_size - 1)
