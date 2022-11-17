import numpy as np
import aesara.tensor as at

def get_posterior(mu1, sd1, mu2, sd2):
    var1, var2 = sd1**2, sd2**2
    return mu1 + (var1/(var1+var2))*(mu2 - mu1), at.sqrt((var1*var2)/(var1+var2))

def get_diff_dist(mu1, sd1, mu2, sd2):
    return mu2 - mu1, at.sqrt(sd1**2+sd2**2)

def cumulative_normal(x, mu, sd, s=at.sqrt(2.)):
#     Cumulative distribution function for the standard normal distribution
    return at.clip(0.5 + 0.5 *
                   at.erf((x - mu) / (sd*s)), 1e-9, 1-1e-9)

def softplus(x): 
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0) 