import aesara.tensor as at

def logistic(x):
    return 1. / (1. + at.exp(-x))