import numpy as np


test_case_1a = [
    np.array([1, 2, 1, 2, 1, 2, 1, 2]),
    ]

test_case_1b = [
    np.array([0, 1, 2, 1, 1, 0, 0, 2, 2, 2, 0]),
    np.array([0, 2, 2, 2, 2, 1, 0, 0, 0, 0, 1])
    ]

test_case_1b = [
    np.array([0, 1, 2, 1, 1, 0, 0, 2, 2, 2, 3, 3, 3]),
    np.array([0, 0, 0, 0, 0]),
    np.array([0, 2, 2, 2, 2, 3, 3, 3, 0, 0, 1])
    ]


registered = {
    "test_case_1a": test_case_1a,
    "test_case_1b": test_case_1b,
}