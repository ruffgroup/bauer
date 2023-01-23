import pandas as pd
import numpy as np
import aesara.tensor as at
import arviz as az
from scipy.special import expit

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

def logistic(x): 
    return expit(x)

def summarize_ppc(ppc, groupby=None):

    if groupby is not None:
        ppc = ppc.groupby(groupby).mean()

    e = ppc.mean(1).to_frame('p_predicted')
    hdi = pd.DataFrame(az.hdi(ppc.T.values), index=ppc.index,
                       columns=['hdi025', 'hdi975'])

    return pd.concat((e, hdi), axis=1)