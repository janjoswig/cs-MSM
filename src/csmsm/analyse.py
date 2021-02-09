import numpy as np
import scipy.linalg


def compute_eig(matrix, **kwargs):
    eigval, eigvec_l, eigvec_r = scipy.linalg.eig(
        matrix, left=True, right=True, **kwargs
        )

    eigval = abs(eigval.real)

    sorted_indices = np.argsort(eigval)[::-1]
    eigval = eigval[sorted_indices]

    eigvec_r = eigvec_r[:, sorted_indices]
    eigvec_l = eigvec_l[:, sorted_indices]

    if all(eigvec_r[0] < 0):
        eigvec_r *= -1
        eigvec_l *= -1

    return eigval, eigvec_l, eigvec_r
