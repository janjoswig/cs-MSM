import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.special


def smoothstep(x, x_min=0, x_max=1, y_min=0, y_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), y_min, y_max)

    result = 0
    for n in range(0, N + 1):
        a = scipy.special.comb(N + n, n)
        b = scipy.special.comb(2 * N + 1, N - n)
        result += a * b * (-x) ** n

    result *= x ** (N + 1)

    return result
