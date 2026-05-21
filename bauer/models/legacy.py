"""Legacy bauer risky-choice models.

Restored from commit 2302d9a (April 2026) for backwards compatibility.
These pre-date the unified safe-vs-risky handling in ``risky_choice.py``;
new code should generally prefer the current ``RiskModel``/``FlexibleNoiseRiskModel``
family. Re-exported from ``bauer.models`` for convenience.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..core import BaseModel, RegressionModel
from ..utils.bayes import get_posterior, cumulative_normal


class SafeVsRiskyModel(BaseModel):
    """Bayesian observer model for risky choice between a safe and a risky option.

    The observer forms posterior beliefs about each option's log-magnitude
    via Bayesian updating of a prior with noisy evidence, then compares
    expected values (incorporating probability) to make a choice.

    Data format:
        - n1, n2: magnitudes (positive for gains, negative for losses)
        - p1, p2: probabilities (risky option has p < 1, safe has p = 1)
        - choice: True = chose the risky option

    Args:
        domain: 'gain' or 'loss' (required)
        separate_priors: separate prior mu/sd for risky vs safe (default True)
        fix_prior_mus: fix prior means to data statistics (default False)
        fix_prior_sds: fix prior sds to 1.0 (default False)
        separate_evidence_sd: separate evidence noise for n1 vs n2 (default True)
    """

    paradigm_keys = ['n1', 'n2', 'p1', 'p2']

    def __init__(self, data=None, domain='gain',
                 separate_priors=True,
                 fix_prior_mus=False, fix_prior_sds=False,
                 separate_evidence_sd=True):

        assert domain in ('gain', 'loss'), "domain must be 'gain' or 'loss'"

        self.domain = domain
        self.separate_priors = separate_priors
        self.fix_prior_mus = fix_prior_mus
        self.fix_prior_sds = fix_prior_sds
        self.separate_evidence_sd = separate_evidence_sd

        super().__init__(data)

    # ── free parameters ──────────────────────────────────────────────

    def get_free_parameters(self):
        free = {}
        mu_spec = {'mu_intercept': np.log(20.), 'sigma_intercept': np.log(20.) / 4., 'transform': 'softplus'}
        sd_spec = {'mu_intercept': 1., 'transform': 'softplus'}
        noise_spec = {'mu_intercept': -1., 'transform': 'softplus'}

        if not self.fix_prior_mus:
            if self.separate_priors:
                free['prior_mu_risky'] = {**mu_spec}
                free['prior_mu_safe'] = {**mu_spec}
            else:
                free['prior_mu'] = {**mu_spec}

        if not self.fix_prior_sds:
            if self.separate_priors:
                free['prior_sd_risky'] = {**sd_spec}
                free['prior_sd_safe'] = {**sd_spec}
            else:
                free['prior_sd'] = {**sd_spec}

        if self.separate_evidence_sd:
            free['evidence_sd_n1'] = {**noise_spec}
            free['evidence_sd_n2'] = {**noise_spec}
        else:
            free['evidence_sd'] = {**noise_spec}

        return free

    # ── model inputs ─────────────────────────────────────────────────

    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()

        n1, n2 = model['n1'], model['n2']
        p1, p2 = model['p1'], model['p2']
        logn1 = pt.log(pt.abs(n1))
        logn2 = pt.log(pt.abs(n2))
        risky_first = (p1 < p2)

        # --- Priors (in role-space: risky / safe) ---
        if self.fix_prior_mus:
            if self.separate_priors:
                risky_n = pt.where(risky_first, n1, n2)
                safe_n = pt.where(risky_first, n2, n1)
                prior_mu_risky = pt.mean(pt.log(pt.abs(risky_n)))
                prior_mu_safe = pt.mean(pt.log(pt.abs(safe_n)))
            else:
                prior_mu_risky = prior_mu_safe = (pt.sum(logn1) + pt.sum(logn2)) / (2 * pt.sum(pt.ones_like(logn1)))
        else:
            if self.separate_priors:
                prior_mu_risky = parameters['prior_mu_risky']
                prior_mu_safe = parameters['prior_mu_safe']
            else:
                prior_mu_risky = prior_mu_safe = parameters['prior_mu']

        if self.fix_prior_sds:
            prior_sd_risky = prior_sd_safe = 1.0
        elif self.separate_priors:
            prior_sd_risky = parameters['prior_sd_risky']
            prior_sd_safe = parameters['prior_sd_safe']
        else:
            prior_sd_risky = prior_sd_safe = parameters['prior_sd']

        # --- Evidence noise (built in position-space, then mapped) ---
        if self.separate_evidence_sd:
            noise1 = parameters['evidence_sd_n1']
            noise2 = parameters['evidence_sd_n2']
        else:
            noise1 = noise2 = parameters['evidence_sd']

        return {
            'prior_mu_risky': prior_mu_risky, 'prior_sd_risky': prior_sd_risky,
            'prior_mu_safe': prior_mu_safe, 'prior_sd_safe': prior_sd_safe,
            'logn1': logn1, 'logn2': logn2,
            'ev_sd1': noise1, 'ev_sd2': noise2,
            'p1': p1, 'p2': p2,
            'risky_first': risky_first,
        }

    # ── choice predictions ───────────────────────────────────────────

    def _get_choice_predictions(self, mi):
        risky_first = mi['risky_first']

        # Map priors: position → role
        mu1 = pt.where(risky_first, mi['prior_mu_risky'], mi['prior_mu_safe'])
        mu2 = pt.where(risky_first, mi['prior_mu_safe'], mi['prior_mu_risky'])
        sd1 = pt.where(risky_first, mi['prior_sd_risky'], mi['prior_sd_safe'])
        sd2 = pt.where(risky_first, mi['prior_sd_safe'], mi['prior_sd_risky'])

        # Bayesian posterior per option (position-space)
        post1_mu, post1_sd = get_posterior(mu1, sd1, mi['logn1'], mi['ev_sd1'])
        post2_mu, post2_sd = get_posterior(mu2, sd2, mi['logn2'], mi['ev_sd2'])

        # Map posteriors: position → role
        risky_post = pt.where(risky_first, post1_mu, post2_mu)
        safe_post = pt.where(risky_first, post2_mu, post1_mu)

        # Decision noise per option: β·ν = post_sd² / evidence_sd
        # (see Khaw, Li & Woodford 2020, eq. 2.5)
        decision_sd1 = post1_sd**2 / mi['ev_sd1']
        decision_sd2 = post2_sd**2 / mi['ev_sd2']
        risky_decision_sd = pt.where(risky_first, decision_sd1, decision_sd2)
        safe_decision_sd = pt.where(risky_first, decision_sd2, decision_sd1)

        # Decision variable: always risky − safe
        diff_mu = risky_post - safe_post
        diff_sd = pt.sqrt(risky_decision_sd**2 + safe_decision_sd**2)

        # Probability threshold
        safe_prob = pt.where(risky_first, mi['p2'], mi['p1'])
        risky_prob = pt.where(risky_first, mi['p1'], mi['p2'])
        threshold = pt.log(safe_prob / risky_prob)

        # Gains: choose risky when perceived risky > safe (accounting for probability)
        #   → p_risky = P(diff > threshold)
        # Losses: choose risky when perceived risky loss isn't too big
        #   → p_risky = P(diff < threshold)
        if self.domain == 'gain':
            p_risky = 1.0 - cumulative_normal(threshold, diff_mu, diff_sd)
        else:
            p_risky = cumulative_normal(threshold, diff_mu, diff_sd)

        return p_risky


class SafeVsRiskyRegressionModel(RegressionModel, SafeVsRiskyModel):
    def __init__(self, data=None, regressors=None, **kwargs):
        SafeVsRiskyModel.__init__(self, data=data, **kwargs)
        RegressionModel.__init__(self, regressors=regressors)

    def get_trialwise_variable(self, key):
        return super().get_trialwise_variable(key)
    

class SafeVsRiskyMemoryModel(SafeVsRiskyModel):
    """Safe-vs-risky model with encoding noise shared across both options
    and an extra working-memory noise term on option 1 only.

    memory_model:
        - 'independent': same as original SafeVsRiskyModel
        - 'shared_perceptual_noise': option1 = encoding + memory, option2 = encoding
    """

    def __init__(self, data=None, domain="gain",
        separate_priors=True, fix_prior_mus=False, fix_prior_sds=False,
        separate_evidence_sd=True, memory_model="shared_perceptual_noise", combine_noise="add_sd"):
        self.memory_model = memory_model
        self.combine_noise = combine_noise

        super().__init__(data=data, domain=domain, separate_priors=separate_priors,
                         fix_prior_mus=fix_prior_mus, fix_prior_sds=fix_prior_sds, separate_evidence_sd=separate_evidence_sd)

    def get_free_parameters(self):
        free = {}
        mu_spec = {"mu_intercept": np.log(20.), "sigma_intercept": np.log(20.) / 4., "transform": "softplus"}
        sd_spec = {"mu_intercept": 1., "transform": "softplus"}
        noise_spec = {"mu_intercept": -1., "transform": "softplus"}

        if not self.fix_prior_mus:
            if self.separate_priors:
                free["prior_mu_risky"] = {**mu_spec}
                free["prior_mu_safe"] = {**mu_spec}
            else:
                free["prior_mu"] = {**mu_spec}

        if not self.fix_prior_sds:
            if self.separate_priors:
                free["prior_sd_risky"] = {**sd_spec}
                free["prior_sd_safe"] = {**sd_spec}
            else:
                free["prior_sd"] = {**sd_spec}

        # evidence / memory parameters
        if self.memory_model == "independent":
            if self.separate_evidence_sd:
                free["evidence_sd_n1"] = {**noise_spec}
                free["evidence_sd_n2"] = {**noise_spec}
            else:
                free["evidence_sd"] = {**noise_spec}

        elif self.memory_model == "shared_perceptual_noise":
            free["encoding_noise_sd"] = {**noise_spec}
            free["memory_noise_sd"] = {**noise_spec}

        else:
            raise ValueError(f"Unknown memory_model: {self.memory_model}")

        return free

    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()

        n1, n2 = model["n1"], model["n2"]
        p1, p2 = model["p1"], model["p2"]
        logn1 = pt.log(pt.abs(n1))
        logn2 = pt.log(pt.abs(n2))
        risky_first = (p1 < p2)

        if self.fix_prior_mus:
            if self.separate_priors:
                risky_n = pt.where(risky_first, n1, n2)
                safe_n = pt.where(risky_first, n2, n1)
                prior_mu_risky = pt.mean(pt.log(pt.abs(risky_n)))
                prior_mu_safe = pt.mean(pt.log(pt.abs(safe_n)))
            else:
                prior_mu_risky = prior_mu_safe = (pt.sum(logn1) + pt.sum(logn2)) / (2 * pt.sum(pt.ones_like(logn1)))
        else:
            if self.separate_priors:
                prior_mu_risky = parameters["prior_mu_risky"]
                prior_mu_safe = parameters["prior_mu_safe"]
            else:
                prior_mu_risky = prior_mu_safe = parameters["prior_mu"]

        if self.fix_prior_sds:
            prior_sd_risky = prior_sd_safe = 1.0
        elif self.separate_priors:
            prior_sd_risky = parameters["prior_sd_risky"]
            prior_sd_safe = parameters["prior_sd_safe"]
        else:
            prior_sd_risky = prior_sd_safe = parameters["prior_sd"]


        if self.memory_model == "independent":
            if self.separate_evidence_sd:
                noise1 = parameters["evidence_sd_n1"]
                noise2 = parameters["evidence_sd_n2"]
            else:
                noise1 = noise2 = parameters["evidence_sd"]

        elif self.memory_model == "shared_perceptual_noise":
            enc = parameters["encoding_noise_sd"]
            mem = parameters["memory_noise_sd"]

            if self.combine_noise == "add_sd":
                noise1 = enc + mem
            elif self.combine_noise == "variance":
                noise1 = pt.sqrt(enc**2 + mem**2)
            else:
                raise ValueError("combine_noise must be 'add_sd' or 'variance'")

            noise2 = enc

        else:
            raise ValueError(f"Unknown memory_model: {self.memory_model}")

        return {"prior_mu_risky": prior_mu_risky, "prior_sd_risky": prior_sd_risky, "prior_mu_safe": prior_mu_safe, "prior_sd_safe": prior_sd_safe,
                "logn1": logn1, "logn2": logn2, "ev_sd1": noise1, "ev_sd2": noise2, "p1": p1, "p2": p2, "risky_first": risky_first}
    


class JointSafeVsRiskyModel(BaseModel):
    """Joint Bayesian observer model for risky choice across gain and loss trials.

    This model fits gain and loss trials together in one likelihood.

    Required data columns:
        - n1, n2: magnitudes, positive for gains and negative for losses
        - p1, p2: probabilities
        - is_gain: 1 for gain trials, 0 for loss trials
        - choice: True = chose risky option

    Main options:
        prior_scope:
            "global":
                one prior shared across gain/loss and risky/safe

            "role":
                risky and safe priors, shared across gain/loss

            "domain":
                gain and loss priors, shared across risky/safe

            "domain_role":
                separate priors for gain/loss × risky/safe
                This gives the 12-parameter model when evidence_scope="domain_position".

        evidence_scope:
            "global":
                one evidence noise parameter

            "position":
                evidence_sd_n1, evidence_sd_n2

            "domain":
                gain_evidence_sd, loss_evidence_sd

            "domain_position":
                gain_evidence_sd_n1, gain_evidence_sd_n2,
                loss_evidence_sd_n1, loss_evidence_sd_n2
    """

    paradigm_keys = ['n1', 'n2', 'p1', 'p2', 'is_gain']

    def __init__(self, data=None,
                 prior_scope="global",
                 evidence_scope="domain_position",
                 fix_prior_mus=False,
                 fix_prior_sds=False):

        assert prior_scope in (
            "global",
            "role",
            "domain",
            "domain_role"
        )

        assert evidence_scope in (
            "global",
            "position",
            "domain",
            "domain_position"
        )

        self.prior_scope = prior_scope
        self.evidence_scope = evidence_scope
        self.fix_prior_mus = fix_prior_mus
        self.fix_prior_sds = fix_prior_sds

        if data is not None:
            data = data.copy()

            if "is_gain" not in data.columns:
                if "domain" in data.columns:
                    data["is_gain"] = (data["domain"] == "gain").astype(int)
                else:
                    raise ValueError(
                        "JointSafeVsRiskyModel requires an 'is_gain' column, "
                        "or a 'domain' column with values 'gain'/'loss'."
                    )

        super().__init__(data)

    # ── free parameters ──────────────────────────────────────────────

    def get_free_parameters(self):
        free = {}

        mu_spec = {
            'mu_intercept': np.log(20.),
            'sigma_intercept': np.log(20.) / 4.,
            'transform': 'softplus'
        }

        sd_spec = {
            'mu_intercept': 1.,
            'transform': 'softplus'
        }

        noise_spec = {
            'mu_intercept': -1.,
            'transform': 'softplus'
        }

        # --- Prior means ---
        if not self.fix_prior_mus:
            if self.prior_scope == "global":
                free["prior_mu"] = {**mu_spec}

            elif self.prior_scope == "role":
                free["prior_mu_risky"] = {**mu_spec}
                free["prior_mu_safe"] = {**mu_spec}

            elif self.prior_scope == "domain":
                free["gain_prior_mu"] = {**mu_spec}
                free["loss_prior_mu"] = {**mu_spec}

            elif self.prior_scope == "domain_role":
                free["gain_prior_mu_risky"] = {**mu_spec}
                free["gain_prior_mu_safe"] = {**mu_spec}
                free["loss_prior_mu_risky"] = {**mu_spec}
                free["loss_prior_mu_safe"] = {**mu_spec}

        # --- Prior SDs ---
        if not self.fix_prior_sds:
            if self.prior_scope == "global":
                free["prior_sd"] = {**sd_spec}

            elif self.prior_scope == "role":
                free["prior_sd_risky"] = {**sd_spec}
                free["prior_sd_safe"] = {**sd_spec}

            elif self.prior_scope == "domain":
                free["gain_prior_sd"] = {**sd_spec}
                free["loss_prior_sd"] = {**sd_spec}

            elif self.prior_scope == "domain_role":
                free["gain_prior_sd_risky"] = {**sd_spec}
                free["gain_prior_sd_safe"] = {**sd_spec}
                free["loss_prior_sd_risky"] = {**sd_spec}
                free["loss_prior_sd_safe"] = {**sd_spec}

        # --- Evidence noise ---
        if self.evidence_scope == "global":
            free["evidence_sd"] = {**noise_spec}

        elif self.evidence_scope == "position":
            free["evidence_sd_n1"] = {**noise_spec}
            free["evidence_sd_n2"] = {**noise_spec}

        elif self.evidence_scope == "domain":
            free["gain_evidence_sd"] = {**noise_spec}
            free["loss_evidence_sd"] = {**noise_spec}

        elif self.evidence_scope == "domain_position":
            free["gain_evidence_sd_n1"] = {**noise_spec}
            free["gain_evidence_sd_n2"] = {**noise_spec}
            free["loss_evidence_sd_n1"] = {**noise_spec}
            free["loss_evidence_sd_n2"] = {**noise_spec}

        return free


    @staticmethod
    def _masked_mean(x, mask):
        """PyTensor-safe masked mean."""
        mask_f = pt.cast(mask, "float64")
        denom = pt.maximum(pt.sum(mask_f), 1.0)
        return pt.sum(pt.where(mask, x, 0.0)) / denom

    def _fixed_prior_mus(self, logn1, logn2, risky_first, is_gain):
        """Compute fixed prior means from the stimulus statistics."""
        risky_logn = pt.where(risky_first, logn1, logn2)
        safe_logn = pt.where(risky_first, logn2, logn1)

        all_logn = pt.concatenate([logn1, logn2])

        if self.prior_scope == "global":
            prior_mu = pt.mean(all_logn)
            return prior_mu, prior_mu

        elif self.prior_scope == "role":
            prior_mu_risky = pt.mean(risky_logn)
            prior_mu_safe = pt.mean(safe_logn)
            return prior_mu_risky, prior_mu_safe

        elif self.prior_scope == "domain":
            gain_mu = self._masked_mean(
                pt.concatenate([logn1, logn2]),
                pt.concatenate([is_gain, is_gain])
            )
            loss_mu = self._masked_mean(
                pt.concatenate([logn1, logn2]),
                pt.concatenate([~is_gain, ~is_gain])
            )
            prior_mu = pt.where(is_gain, gain_mu, loss_mu)
            return prior_mu, prior_mu

        elif self.prior_scope == "domain_role":
            gain_mu_risky = self._masked_mean(risky_logn, is_gain)
            gain_mu_safe = self._masked_mean(safe_logn, is_gain)
            loss_mu_risky = self._masked_mean(risky_logn, ~is_gain)
            loss_mu_safe = self._masked_mean(safe_logn, ~is_gain)

            prior_mu_risky = pt.where(
                is_gain,
                gain_mu_risky,
                loss_mu_risky
            )
            prior_mu_safe = pt.where(
                is_gain,
                gain_mu_safe,
                loss_mu_safe
            )
            return prior_mu_risky, prior_mu_safe


    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()

        n1, n2 = model["n1"], model["n2"]
        p1, p2 = model["p1"], model["p2"]

        is_gain = pt.gt(model["is_gain"], 0.5)

        logn1 = pt.log(pt.abs(n1))
        logn2 = pt.log(pt.abs(n2))

        risky_first = p1 < p2


        if self.fix_prior_mus:
            prior_mu_risky, prior_mu_safe = self._fixed_prior_mus(
                logn1=logn1,
                logn2=logn2,
                risky_first=risky_first,
                is_gain=is_gain
            )

        else:
            if self.prior_scope == "global":
                prior_mu_risky = parameters["prior_mu"]
                prior_mu_safe = parameters["prior_mu"]

            elif self.prior_scope == "role":
                prior_mu_risky = parameters["prior_mu_risky"]
                prior_mu_safe = parameters["prior_mu_safe"]

            elif self.prior_scope == "domain":
                prior_mu = pt.where(
                    is_gain,
                    parameters["gain_prior_mu"],
                    parameters["loss_prior_mu"]
                )
                prior_mu_risky = prior_mu
                prior_mu_safe = prior_mu

            elif self.prior_scope == "domain_role":
                prior_mu_risky = pt.where(
                    is_gain,
                    parameters["gain_prior_mu_risky"],
                    parameters["loss_prior_mu_risky"]
                )
                prior_mu_safe = pt.where(
                    is_gain,
                    parameters["gain_prior_mu_safe"],
                    parameters["loss_prior_mu_safe"]
                )

        if self.fix_prior_sds:
            prior_sd_risky = 1.0
            prior_sd_safe = 1.0

        else:
            if self.prior_scope == "global":
                prior_sd_risky = parameters["prior_sd"]
                prior_sd_safe = parameters["prior_sd"]

            elif self.prior_scope == "role":
                prior_sd_risky = parameters["prior_sd_risky"]
                prior_sd_safe = parameters["prior_sd_safe"]

            elif self.prior_scope == "domain":
                prior_sd = pt.where(
                    is_gain,
                    parameters["gain_prior_sd"],
                    parameters["loss_prior_sd"]
                )
                prior_sd_risky = prior_sd
                prior_sd_safe = prior_sd

            elif self.prior_scope == "domain_role":
                prior_sd_risky = pt.where(
                    is_gain,
                    parameters["gain_prior_sd_risky"],
                    parameters["loss_prior_sd_risky"]
                )
                prior_sd_safe = pt.where(
                    is_gain,
                    parameters["gain_prior_sd_safe"],
                    parameters["loss_prior_sd_safe"]
                )


        if self.evidence_scope == "global":
            ev_sd1 = parameters["evidence_sd"]
            ev_sd2 = parameters["evidence_sd"]

        elif self.evidence_scope == "position":
            ev_sd1 = parameters["evidence_sd_n1"]
            ev_sd2 = parameters["evidence_sd_n2"]

        elif self.evidence_scope == "domain":
            ev_sd = pt.where(
                is_gain,
                parameters["gain_evidence_sd"],
                parameters["loss_evidence_sd"]
            )
            ev_sd1 = ev_sd
            ev_sd2 = ev_sd

        elif self.evidence_scope == "domain_position":
            ev_sd1 = pt.where(
                is_gain,
                parameters["gain_evidence_sd_n1"],
                parameters["loss_evidence_sd_n1"]
            )
            ev_sd2 = pt.where(
                is_gain,
                parameters["gain_evidence_sd_n2"],
                parameters["loss_evidence_sd_n2"]
            )

        return {
            "prior_mu_risky": prior_mu_risky,
            "prior_sd_risky": prior_sd_risky,
            "prior_mu_safe": prior_mu_safe,
            "prior_sd_safe": prior_sd_safe,
            "logn1": logn1,
            "logn2": logn2,
            "ev_sd1": ev_sd1,
            "ev_sd2": ev_sd2,
            "p1": p1,
            "p2": p2,
            "risky_first": risky_first,
            "is_gain": is_gain,
        }

    # ── choice predictions ───────────────────────────────────────────

    def _get_choice_predictions(self, mi):
        risky_first = mi["risky_first"]

        mu1 = pt.where(
            risky_first,
            mi["prior_mu_risky"],
            mi["prior_mu_safe"]
        )
        mu2 = pt.where(
            risky_first,
            mi["prior_mu_safe"],
            mi["prior_mu_risky"]
        )

        sd1 = pt.where(
            risky_first,
            mi["prior_sd_risky"],
            mi["prior_sd_safe"]
        )
        sd2 = pt.where(
            risky_first,
            mi["prior_sd_safe"],
            mi["prior_sd_risky"]
        )

        post1_mu, post1_sd = get_posterior(
            mu1,
            sd1,
            mi["logn1"],
            mi["ev_sd1"]
        )
        post2_mu, post2_sd = get_posterior(
            mu2,
            sd2,
            mi["logn2"],
            mi["ev_sd2"]
        )

        risky_post = pt.where(risky_first, post1_mu, post2_mu)
        safe_post = pt.where(risky_first, post2_mu, post1_mu)

        decision_sd1 = post1_sd**2 / mi["ev_sd1"]
        decision_sd2 = post2_sd**2 / mi["ev_sd2"]

        risky_decision_sd = pt.where(
            risky_first,
            decision_sd1,
            decision_sd2
        )
        safe_decision_sd = pt.where(
            risky_first,
            decision_sd2,
            decision_sd1
        )

        diff_mu = risky_post - safe_post
        diff_sd = pt.sqrt(risky_decision_sd**2 + safe_decision_sd**2)

        # Probability threshold.
        safe_prob = pt.where(risky_first, mi["p2"], mi["p1"])
        risky_prob = pt.where(risky_first, mi["p1"], mi["p2"])
        threshold = pt.log(safe_prob / risky_prob)

        p_risky_gain = 1.0 - cumulative_normal(
            threshold,
            diff_mu,
            diff_sd
        )

        p_risky_loss = cumulative_normal(
            threshold,
            diff_mu,
            diff_sd
        )

        p_risky = pt.where(
            mi["is_gain"],
            p_risky_gain,
            p_risky_loss
        )

        return p_risky
    

