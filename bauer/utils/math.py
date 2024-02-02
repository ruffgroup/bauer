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

def logit_np(p): return np.log(p / (1 - p))

def logit(p):
    return pt.log(p / (1 - p))

def logit_derivative(p):
    return 1 / (p-p**2)

def gaussian_pdf(x, mean, std):
    variance = std**2
    exponent = -((x - mean)**2) / (2 * variance)
    normalization = 1 / (std * pt.sqrt(2 * np.pi))
    return normalization * pt.exp(exponent)