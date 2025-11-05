import numpy as np


DISCOUNT_FACTOR = 0.9
HIDDEN_SIZE = 192
BASE_STATE_SIZE = 60
ALPHA = 0.3


def get_weighted_central_moment(n: int, weights, norms_squared):
    exponent = n / 2
    numerator = min((weights * norms_squared**exponent).sum(), 1e8)
    inertia_denom_w = np.linalg.norm(weights)
    inertia_denom_n = np.linalg.norm(norms_squared**exponent)
    return numerator / max(1e-5, inertia_denom_w * max(1e-5, inertia_denom_n))
