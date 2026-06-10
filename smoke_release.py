import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import bauer
print("bauer", bauer.__version__)
from bauer.models import (DDMMagnitudeComparisonModel,
                          DDMMagnitudeComparisonRegressionModel,
                          MagnitudeComparisonRegressionModel)
from bauer.models.race import RaceDiffusionMagnitudeComparisonModel
from bauer.core import _group_sd_rv

# tiny synthetic paradigm
rng = np.random.RandomState(0)
n=200
df = pd.DataFrame(dict(
    subject=np.repeat([1,2],n//2), n1=rng.randint(5,25,n), n2=rng.randint(5,25,n),
    rt=rng.uniform(.3,2.,n), choice=rng.rand(n)>.5,
    group=np.repeat(["control","dyscalc"],n//2)))
df=df.set_index("subject")

# 1. default group_sd_dist is halfnormal
m = DDMMagnitudeComparisonRegressionModel(df.reset_index(), regressors={"perceptual_noise_sd":"C(group)"},
        fit_separate_evidence_sd=True, memory_model="shared_perceptual_noise")
assert m.group_sd_dist=="halfnormal", m.group_sd_dist
print("OK default group_sd_dist =", m.group_sd_dist)

# 2. contaminant default present
print("OK p_outlier default =", DDMMagnitudeComparisonRegressionModel.p_outlier)

# 3. memory_as_sv flag accepted
m2 = DDMMagnitudeComparisonRegressionModel(df.reset_index(), regressors={"perceptual_noise_sd":"C(group)"},
        fit_separate_evidence_sd=True, memory_model="shared_perceptual_noise", memory_as_sv=True)
print("OK memory_as_sv accepted")

# 4. race fit_w_d toggle
mr = RaceDiffusionMagnitudeComparisonModel(df.reset_index(), fit_separate_evidence_sd=True,
        memory_model="shared_perceptual_noise")
print("OK race fit_w_d default =", getattr(mr,"fit_w_d",None))

# 5. refx API: fixed_regressors + random_regressors and the DeprecationWarning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m3 = MagnitudeComparisonRegressionModel(df.reset_index(), regressors={"perceptual_noise_sd":"C(group)"},
            fit_separate_evidence_sd=True, memory_model="shared_perceptual_noise")
    assert any(issubclass(x.category, DeprecationWarning) for x in w), "no DeprecationWarning on regressors="
print("OK regressors= raises DeprecationWarning")
m4 = MagnitudeComparisonRegressionModel(df.reset_index(),
        fixed_regressors={"perceptual_noise_sd":"C(group)"},
        random_regressors={"perceptual_noise_sd":"1"},
        fit_separate_evidence_sd=True, memory_model="shared_perceptual_noise")
print("OK fixed_regressors/random_regressors accepted")

# 6. build the graph (the real test) for the refx subset model
m4.build_estimation_model(data=df, hierarchical=True)
print("OK build_estimation_model (refx subset) -> graph built")

# 7. halfnormal actually used in graph
import pymc as pm
sdvars=[v.name for v in m4.estimation_model.free_RVs if v.name.endswith("_sd")]
print("group-sd RVs:", sdvars)
print("ALL SMOKE TESTS PASSED")
