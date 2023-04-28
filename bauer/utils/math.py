import numpy as np
import aesara.tensor as at

def logistic(x):
    return 1. / (1. + at.exp(-x))

def inverse_softplus(x):
    return at.log(at.exp(x) - 1.)


def softplus_np(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
def logistic_np(x): return 1 / (1 + np.exp(-x))
