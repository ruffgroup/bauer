import numpy as np
import pytensor.tensor as pt

def logistic(x):
    return 1. / (1. + pt.exp(-x))

def inverse_softplus(x):
    return pt.log(pt.exp(x) - 1.)


def softplus_np(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def inverse_softplus_np(x):
    return np.log(np.exp(x) - 1.)

def logistic_np(x): return 1 / (1 + np.exp(-x))
