import aesara.tensor as at

def logistic(x):
    return 1. / (1. + at.exp(-x))

def inverse_softplus(x):
    return at.log(at.exp(x) - 1.)