import functools
import warnings
import pandas as pd
import pymc as pm
import numpy as np
from .utils.bayes import cumulative_normal, get_diff_dist, get_posterior
from .utils.math import logistic, softplus_np, logistic_np, logit_np, inverse_softplus_np
import pytensor.tensor as pt
from patsy import dmatrix, build_design_matrices


# Mapping of deprecated (old) __init__ keyword names -> their current names.
# Add a new entry here to retire another misspelled / renamed kwarg; the
# translation + DeprecationWarning are handled automatically for every
# BaseModel subclass via __init_subclass__ (and for BaseModel itself).
_DEPRECATED_INIT_KWARGS = {
    'fit_seperate_evidence_sd': 'fit_separate_evidence_sd',
}


def _translate_deprecated_kwargs(func):
    """Wrap an ``__init__`` so deprecated kwarg names are translated.

    Emits a :class:`DeprecationWarning` and forwards the value under the new
    name. If both the old and new names are supplied, the old one is dropped
    and the new one wins. The wrapper is idempotent (re-wrapping is a no-op)
    so cooperative multiple inheritance doesn't double-wrap, and because the
    outer ``__init__`` already renames before calling ``super().__init__``,
    inner ``__init__``s only ever see the new name (no double-warning).
    """
    if getattr(func, '_bauer_alias_wrapped', False):
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for old, new in _DEPRECATED_INIT_KWARGS.items():
            if old in kwargs:
                warnings.warn(
                    f"'{old}' is deprecated and will be removed; use '{new}'.",
                    DeprecationWarning, stacklevel=2,
                )
                if new in kwargs:
                    # Both passed: keep the new spelling, drop the old.
                    kwargs.pop(old)
                else:
                    kwargs[new] = kwargs.pop(old)
        return func(*args, **kwargs)

    wrapper._bauer_alias_wrapped = True
    return wrapper


def _group_sd_rv(name, scale, dist='halfcauchy', dims=None):
    """Group-level SD prior for a hierarchical node.

    ``'halfcauchy'`` (default, ``HalfCauchy(beta=scale)``) is bauer's historical
    choice but has an infinite-variance heavy tail: a poorly-identified group SD
    can wander to huge values, producing a funnel and divergences (seen e.g. on
    DDM noise components and near-0 lapse rates). ``'halfnormal'``
    (``HalfNormal(sigma=scale)``) has a light tail that tames the SD and removes
    that pathology, at the cost of mild over-shrinkage if true between-subject
    heterogeneity is large.
    """
    if dist == 'halfnormal':
        return pm.HalfNormal(name, sigma=scale, dims=dims)
    if dist == 'halfcauchy':
        return pm.HalfCauchy(name, beta=scale, dims=dims)
    raise ValueError(
        f"group_sd_dist must be 'halfcauchy' or 'halfnormal', got {dist!r}")


class BaseModel(object):

    paradigm_keys = []

    # Group-level SD prior family for hierarchical nodes: 'halfcauchy'
    # (default, historical) or 'halfnormal' (tamer tail; avoids the group-SD
    # funnel that drives divergences when between-subject variance is weakly
    # identified). See _group_sd_rv. Set per-instance/subclass to override.
    group_sd_dist = 'halfcauchy'

    # Per-model-class hints for the NUTS sampler.
    #
    # ``recommended_nuts_kwargs`` is forwarded to numpyro / blackjax via
    # ``pm.sampling.jax.sample_*_nuts(nuts_kwargs=...)``. DDM/RDM mixins
    # override this with ``{'dense_mass': True}`` because the posterior has
    # strongly-correlated parameters (v_scale × evidence_sd × a) that
    # diagonal mass cannot navigate (step-size collapses, tree-depth maxes,
    # chains get stuck at numerical singularities).
    #
    # ``recommended_pymc_init`` is the equivalent for the default pymc
    # backend — passed as ``pm.sample(init=...)``. Both 'adapt_full' and
    # 'jitter+adapt_full' enable full mass-matrix adaptation. Default
    # ``None`` = use pymc's default ('auto' → 'jitter+adapt_diag').
    #
    # Empty / None defaults = use whatever the backend chooses.
    recommended_nuts_kwargs: dict = {}
    recommended_pymc_init: str | None = None

    # ``recommended_init`` enables bauer's starting-point finder by default for
    # this model class (see ``BaseModel.get_initial_points`` and ``sample``).
    # DDM/RDM mixins set this to 'mapjitter' because their posteriors are nasty
    # enough that pymc's generic jitter init makes convergence a seed lottery
    # (verified: vectorized regression-DDM converges ~1/7 seeds without it).
    # ``None`` = use the backend's default init.
    recommended_init: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Wrap each subclass's own __init__ (if it defines one) so the
        # deprecated kwarg translation runs before the real body. Mixins /
        # diamond inheritance are fine: the outermost __init__ renames first,
        # so inner __init__s see only the new name.
        if '__init__' in cls.__dict__:
            cls.__init__ = _translate_deprecated_kwargs(cls.__dict__['__init__'])

    def __init__(self, paradigm, save_trialwise_n_estimates=False):
        """
        data should contain ['n1', 'n2'] and 'choice'.
        The latter should be a boolean that indicates whether the *second*
        option was chosen.

        """
        self.paradigm = paradigm
        self.save_trialwise_n_estimates = save_trialwise_n_estimates
        self.free_parameters = self.get_free_parameters()

    def _get_paradigm(self, paradigm=None, subject_mapping=None):

        if paradigm is None:
            paradigm = self.data

        paradigm_ = {}
        paradigm_['_data_n'] = len(paradigm)

        for key in self.paradigm_keys:
            paradigm_[key] = paradigm[key].values

        if subject_mapping is None:
            if 'subject' in paradigm.index.names:
                paradigm_['subject_ix'], _ = pd.factorize(paradigm.index.get_level_values('subject'))
            elif 'subject' in paradigm.columns:
                paradigm_['subject_ix'], _ = pd.factorize(paradigm['subject'])
        else:
            # Make sure all subjects in the paradigm are in the subject_mapping
            new_subject_ids = paradigm['subject'].unique()
            assert (np.array_equal(new_subject_ids, list(subject_mapping.keys()))), "The unique subjects in the paradigm do not match the subjects in the subject_mapping."
            if 'subject' in paradigm.index.names:
                paradigm_['subject_ix'] = [subject_mapping[subject] for subject in paradigm.index.get_level_values('subject')]
            else:
                paradigm_['subject_ix'] = [subject_mapping[subject] for subject in paradigm['subject']]

        if 'choice' in paradigm.columns:
            paradigm_['choice'] = paradigm['choice'].values

        return paradigm_

    def _get_choice_predictions(self, model_inputs):
        if getattr(self, 'flat_observer_prior', False):
            # Flat improper prior → posterior is the likelihood. No shrinkage.
            post_n1_mu = model_inputs['n1_evidence_mu']
            post_n2_mu = model_inputs['n2_evidence_mu']
        else:
            post_n1_mu, _ = get_posterior(model_inputs['n1_prior_mu'],
                                          model_inputs['n1_prior_sd'],
                                          model_inputs['n1_evidence_mu'],
                                          model_inputs['n1_evidence_sd'])
            post_n2_mu, _ = get_posterior(model_inputs['n2_prior_mu'],
                                          model_inputs['n2_prior_sd'],
                                          model_inputs['n2_evidence_mu'],
                                          model_inputs['n2_evidence_sd'])

        diff_mu, diff_sd = get_diff_dist(post_n2_mu, model_inputs['n2_evidence_sd'], post_n1_mu, model_inputs['n1_evidence_sd'])

        if self.save_trialwise_n_estimates:
            pm.Deterministic('n1_hat', post_n1_mu)
            pm.Deterministic('n2_hat', post_n2_mu)

        return cumulative_normal(model_inputs['threshold'], diff_mu, diff_sd)

    def build_likelihood(self, parameters, save_p_choice=False):
        model = pm.Model.get_context()
        model_inputs = self.get_model_inputs(parameters)

        if save_p_choice:
            p = pm.Deterministic('p', var=self._get_choice_predictions(model_inputs))
        else:
            p = self._get_choice_predictions(model_inputs)

        pm.Bernoulli('ll_bernoulli', p=p, observed=model['choice'])

    def get_parameter_values(self):

        parameters = {}

        for key in self.free_parameters.keys():
            parameters[key] = self.get_trialwise_variable(key)

        return parameters

    def build_estimation_model(self, data=None, coords=None, hierarchical=True, save_p_choice=False, flat_prior=False, paradigm=None):

        # `data` and `paradigm` are accepted synonyms (no deprecation); data wins if both given.
        if data is None:
            data = paradigm if paradigm is not None else self.paradigm

        if not hierarchical and 'subject' in getattr(data, 'index', pd.Index([])).names:
            n_subjects = data.index.get_level_values('subject').nunique()
            if n_subjects > 1:
                raise ValueError(
                    f"hierarchical=False with {n_subjects} subjects in the data. "
                    f"This fits a single shared parameter set (complete pooling), which is almost "
                    f"certainly not what you want. Use hierarchical=True, or pass single-subject "
                    f"data (e.g. data.xs(subject, level='subject')). "
                    f"For individual MAP per subject, use fit_map_individual()."
                )

        if hierarchical and (coords is None):
            assert ('subject' in data.index.names), "Hierarchical estimation requires a multi-index with a 'subject' level."
            coords = {'subject': data.index.unique(level='subject')}

        with pm.Model(coords=coords) as self.estimation_model:
            paradigm = self._get_paradigm(paradigm=data)
            self.set_paradigm(paradigm)
            self.build_priors(hierarchical=hierarchical, flat_prior=flat_prior)
            parameters = self.get_parameter_values()
            self.build_likelihood(parameters, save_p_choice=save_p_choice)

    def build_priors(self, hierarchical=True, flat_prior=False):

        if hierarchical:
            for key, info in self.free_parameters.items():
                self.build_hierarchical_nodes(key, **info)
        else:
            for key, info in self.free_parameters.items():
                self.build_prior(key, flat_prior=flat_prior, **info)

    def build_prior(self, name, mu_intercept=None, sigma_intercept=None, transform='identity',
                    flat_prior=False, beta_mu_mean=0.05, beta_mu_kappa=8.0, **kwargs):

        model = pm.Model.get_context()

        if flat_prior:
            # Flat in the NATURAL (transformed) space — no Jacobian issues.
            # Safe for both MAP (= true MLE) and sampling.
            if transform == 'identity':
                pm.Flat(name)
            elif transform == 'softplus':
                pm.HalfFlat(name)
            elif transform in ('logistic', 'beta'):
                pm.Uniform(name, lower=0, upper=1)
            else:
                raise NotImplementedError
            return

        if transform == 'beta':
            # Single-subject Beta prior with the same small-mean / heavy-tail
            # shape used by the hierarchical 'beta' group prior (see
            # build_hierarchical_nodes). Concentration = beta_mu_kappa.
            a0 = beta_mu_mean * beta_mu_kappa
            b0 = (1.0 - beta_mu_mean) * beta_mu_kappa
            pm.Beta(name, alpha=a0, beta=b0)
            return

        if mu_intercept is None:
            mu_intercept = 0.0
        if sigma_intercept is None:
            sigma_intercept = .5

        if transform == 'identity':
            pm.Normal(name, mu=mu_intercept, sigma=sigma_intercept)
        elif transform == 'softplus':
            pm.Normal(name + '_untransformed', mu=mu_intercept, sigma=sigma_intercept)
            pm.Deterministic(name, var=pt.softplus(model[name + '_untransformed']))
        elif transform == 'logistic':
            pm.Normal(name + '_untransformed', mu=mu_intercept, sigma=sigma_intercept)
            pm.Deterministic(name, var=logistic(model[name + '_untransformed']))
        else:
            raise NotImplementedError

    def set_paradigm(self, paradigm=None):
        for key, value in paradigm.items():
            pm.Data(key, value)

    def update_paradigm(self, paradigm):
        raise NotImplementedError

    def build_prediction_model(self, paradigm, parameters,):

        assert (isinstance(parameters, dict) or isinstance(parameters, pd.DataFrame)), "Parameters should be a dictionary or a DataFrame."

        if paradigm is None:
            paradigm = self.data

        if isinstance(parameters, pd.DataFrame):
            parameter_subjects = parameters['subject'] if 'subject' in parameters else parameters.index.get_level_values('subject')

            # Make sure that the unique levels in 'subject' (either index or column) of the paradigm align with the subjects in the parameters
            if 'subject' in paradigm.index.names:
                assert (np.array_equal(paradigm.index.unique(level='subject'), parameter_subjects)), "The unique subjects in the paradigm do not match the subjects in the parameters."
            elif 'subject' in paradigm.columns:
                assert (np.array_equal(paradigm.subject.unique(), parameter_subjects)), "The unique subjects in the paradigm do not match the subjects in the parameters."

            parameters = parameters.to_dict(orient='list')

        with pm.Model() as self.prediction_model:
            paradigm = self._get_paradigm(paradigm=paradigm)
            self.set_paradigm(paradigm)

            # Make parameters flexible
            for key, value in parameters.items():
                pm.Data(key, value)

            parameters = self.get_parameter_values()
            model_inputs = self.get_model_inputs(parameters)

            p = pm.Deterministic('p', var=self._get_choice_predictions(model_inputs))
            pm.Bernoulli('ll_bernoulli', p=p)

    def predict(self, paradigm, parameters):

        self.build_prediction_model(paradigm, parameters)
        data = self.prediction_model['p'].eval()
        data = pd.DataFrame(data, index=paradigm.index, columns=['p_choice'])
        data = data.join(paradigm)

        return data

    def simulate(self, paradigm, parameters, n_samples=1):

        self.build_prediction_model(paradigm, parameters)

        with self.prediction_model:
            data = pm.draw(self.prediction_model['ll_bernoulli'], draws=n_samples)

        if n_samples == 1:
            data = data[np.newaxis, :]

        if not paradigm.index.name:
            paradigm.index.name = 'trial'

        data = pd.DataFrame(data.T, index=paradigm.index, columns=pd.Index(np.arange(n_samples) + 1, name='sample'))
        data = data.stack().to_frame('simulated_choice').astype(bool)
        data = data.join(paradigm)

        if ('subject' in paradigm.columns):
            data = data.set_index('subject', append=True)
            data = data.reorder_levels(['subject'] + list(data.index.names)[:-1])

        return data

    def sample(self, draws=1000, tune=1000, target_accept=0.8, chains=4,
               backend='pymc', find_init=None, **kwargs):
        """Sample from the posterior using the requested NUTS backend.

        Parameters
        ----------
        backend : {'pymc', 'numpyro', 'blackjax'}
            'pymc' uses :func:`pm.sample` (default). 'numpyro' / 'blackjax'
            use the corresponding JAX-NUTS sampler from
            :mod:`pm.sampling.jax`. JAX backends are much faster on GPU.
        chains : int
            Number of chains. Default 4.
        target_accept : float
            NUTS target acceptance probability. 0.8 is fine for well-behaved
            models; 0.95 for hierarchical / DDM-like ones; 0.99 only if
            divergences persist.
        draws, tune : int
            Posterior and warmup draws per chain.
        find_init : {None, 'mapjitter', 'priorjitter', 'pathfinder'}
            Starting-point strategy. ``None`` uses the class default
            (``recommended_init``; ``'mapjitter'`` for DDM/Race). ``'mapjitter'``
            = MAP centre + prior-scaled jitter; ``'priorjitter'`` = prior-central
            centre + jitter; ``'pathfinder'`` = seed each chain from a multipath
            Pathfinder draw (variational; lands in the typical set — best for
            nasty high-dimensional DDM geometries, needs ``pymc_extras``, falls
            back to ``'mapjitter'`` if unavailable). Ignored if ``initvals`` is
            passed in ``**kwargs``.
        **kwargs
            Forwarded to the underlying sampler. Notably:
            - pymc backend: ``init=`` (e.g. 'jitter+adapt_full' for dense
              mass adaptation), ``cores=``, ``random_seed=``.
            - JAX backends: ``nuts_kwargs={'dense_mass': True}``,
              ``chain_method='vectorized'``, ``random_seed=``.

        Auto-applied defaults
        ---------------------
        Subclasses can declare strongly-correlated posteriors via
        ``recommended_pymc_init`` and ``recommended_nuts_kwargs``; this
        method applies them unless the user passes the corresponding kwarg
        explicitly. DDMMixin / RaceMixin do this for full mass-matrix
        adaptation.
        """
        # Starting-point finder. ``find_init`` (or the class default
        # ``recommended_init``) builds dispersed per-chain initial values
        # around a data-informed plausible centre (see get_initial_points),
        # instead of relying on the backend's generic jitter init. This is
        # what stops hard DDM/regression posteriors from being a seed
        # lottery. Skipped if the user supplied their own ``initvals``.
        #
        # ``find_init='pathfinder'`` runs multipath Pathfinder (variational;
        # lands in the typical set rather than at the MAP mode) and seeds each
        # chain from a distinct Pathfinder draw — the right lever for nasty
        # high-dimensional DDM geometries where MAP is the wrong target. See
        # get_initial_points / _pathfinder_initial_points.
        if find_init is None:
            find_init = self.recommended_init
        if find_init and 'initvals' not in kwargs:
            seed = kwargs.get('random_seed', None)
            seed = seed if isinstance(seed, int) else None
            if find_init == 'pathfinder':
                kwargs['initvals'] = self._pathfinder_initial_points(
                    chains=chains, seed=seed)
            else:
                kwargs['initvals'] = self.get_initial_points(
                    chains=chains, use_map=(find_init != 'priorjitter'),
                    seed=seed)

        if backend == 'pymc':
            # When we supply our own dispersed initvals, avoid a jitter-adding
            # init method (we already jittered); else keep the recommendation.
            default_init = getattr(self, 'recommended_pymc_init', None) or 'auto'
            if 'initvals' in kwargs and isinstance(default_init, str):
                default_init = default_init.replace('jitter+', '')
            kwargs.setdefault('init', default_init)
            with self.estimation_model:
                self.idata = pm.sample(
                    draws, tune=tune, chains=chains,
                    target_accept=target_accept,
                    return_inferencedata=True, **kwargs,
                )
        elif backend in ('numpyro', 'blackjax'):
            from pymc.sampling.jax import (
                sample_numpyro_nuts, sample_blackjax_nuts,
            )
            sampler = (sample_numpyro_nuts if backend == 'numpyro'
                       else sample_blackjax_nuts)
            # Merge model recommendation with any explicit override
            rec = dict(getattr(self, 'recommended_nuts_kwargs', {}))
            user = kwargs.pop('nuts_kwargs', None) or {}
            nuts_kwargs = {**rec, **user}
            kwargs.setdefault('chain_method', 'vectorized')
            if 'initvals' in kwargs:
                kwargs.setdefault('jitter', False)  # initvals already dispersed
            with self.estimation_model:
                self.idata = sampler(
                    draws=draws, tune=tune, chains=chains,
                    target_accept=target_accept,
                    nuts_kwargs=nuts_kwargs or None,
                    **kwargs,
                )
        else:
            raise ValueError(f"backend must be 'pymc'/'numpyro'/'blackjax', got {backend!r}")
        return self.idata

    def get_initial_points(self, chains=4, jitter_frac=0.1, use_map=True,
                           n_prior=500, seed=None):
        """Dispersed per-chain starting points for the sampler.

        Returns a list of ``chains`` initval dicts (keyed by the model's
        free-RV value-variable names). Each is a *plausible centre* plus
        per-parameter Gaussian jitter whose SD is ``jitter_frac`` times that
        parameter's prior SD. Chains are dispersed around the centre — never
        all placed *at* it — so the mode (which is not in the typical set) is
        not the start, and between-chain r̂ stays meaningful.

        centre
            ``find_MAP`` (the data-informed posterior mode / plausible value)
            when ``use_map``; otherwise the model's prior-central
            ``initial_point()``. Falls back to ``initial_point()`` if MAP fails.
        jitter scale
            Each free parameter is a plain unconstrained Normal in bauer's
            models (softplus/logistic links are applied downstream as
            Deterministics), so prior draws live in the same space as the
            initvals and their SD is a natural, safe jitter scale.

        Mirrors the init strategy HSSM uses (curated centre + small jitter) —
        which bauer otherwise omits, leaving convergence of hard posteriors a
        seed lottery.
        """
        rng = np.random.default_rng(seed)
        with self.estimation_model:
            ip = self.estimation_model.initial_point()
            centre = {k: np.asarray(v, dtype='float64') for k, v in ip.items()}
            if use_map:
                try:
                    mp = pm.find_MAP(progressbar=False)
                    for k in centre:
                        if k in mp:
                            centre[k] = np.asarray(mp[k], dtype='float64')
                except Exception as e:  # pragma: no cover - robustness
                    warnings.warn(f"find_MAP failed ({e}); using prior-central "
                                  "centre for initial points.")
            try:
                pr = pm.sample_prior_predictive(
                    draws=n_prior, var_names=list(centre), random_seed=seed)
                psd = {k: np.asarray(pr.prior[k]).std(axis=(0, 1))
                       for k in centre if k in pr.prior}
            except Exception:  # pragma: no cover - robustness
                psd = {}

        points = []
        for _ in range(chains):
            pt = {}
            for k in centre:
                scale = np.broadcast_to(np.asarray(psd.get(k, 1.0)), centre[k].shape)
                scale = np.where(scale > 0, scale, 1.0)
                jit = rng.normal(0.0, jitter_frac * scale, size=centre[k].shape)
                pt[k] = (centre[k] + jit).astype(ip[k].dtype)
            points.append(pt)
        return points

    def _pathfinder_initial_points(self, chains=4, seed=None,
                                   num_paths=8, **pathfinder_kwargs):
        """Pathfinder-based dispersed per-chain starting points.

        Runs **multipath Pathfinder** (a variational method that lands in the
        posterior's *typical set*, not at the MAP mode) on the built
        ``estimation_model`` via :func:`pymc_extras.inference.fit`, then turns
        ``chains`` of its draws into per-chain ``initvals`` dicts on the model's
        **value-variable (unconstrained) scale** — the exact format
        :meth:`get_initial_points` returns (keyed by ``model.initial_point()``
        names, e.g. ``a_mu_untransformed``, ``s_log__``, ``p_logodds__``), so it
        is a drop-in replacement as a sampler ``initvals``.

        This is the right init lever for hard, high-dimensional DDM/RDM
        posteriors: MAP (the ``'mapjitter'`` centre) is in the wrong place in
        high dimensions (the mode is not in the typical set), whereas
        Pathfinder draws already sit where the chains should warm up.

        Robustness
        ----------
        If ``pymc_extras`` is missing, or Pathfinder errors / returns too few
        usable draws, this falls back to :meth:`get_initial_points`
        (MAP + jitter) with a :class:`UserWarning` — so a fit never crashes for
        want of the optional dependency.

        Parameters
        ----------
        chains : int
            Number of per-chain initval dicts to return.
        seed : int or None
            Seed forwarded to Pathfinder for reproducibility.
        num_paths : int
            Number of Pathfinder paths (multipath). Default 8.
        **pathfinder_kwargs
            Extra keyword arguments forwarded to
            ``pymc_extras.inference.fit(method='pathfinder', ...)``.
        """
        def _fallback(reason):
            warnings.warn(
                f"Pathfinder init unavailable ({reason}); "
                "falling back to mapjitter initial points.")
            return self.get_initial_points(chains=chains, use_map=True,
                                           seed=seed)

        try:
            from pymc_extras.inference import fit as pmx_fit
        except Exception as e:  # pragma: no cover - optional dependency
            return _fallback(f"could not import pymc_extras ({e})")

        # Draw at least as many Pathfinder samples as we need chains, with a
        # comfortable margin so we can pick well-separated draws.
        num_draws = pathfinder_kwargs.pop('num_draws',
                                          max(1000, 50 * chains))

        # Seed Pathfinder's OWN optimization from a plausible, data-informed
        # point (the MAP) so its paths start in the valid region. Without this,
        # on nasty DDM/RDM geometries Pathfinder can wander into the WFPT's
        # flat/-inf zone and hand NUTS a frozen, all-divergent starting point
        # (observed: independent-noise DDM -> 4000/4000 divergences, within-
        # chain SD 0). If MAP fails, fall back to Pathfinder's default init.
        seed_initvals = pathfinder_kwargs.pop('initvals', None)
        if seed_initvals is None:
            try:
                with self.estimation_model:
                    seed_initvals = pm.find_MAP(progressbar=False)
            except Exception:  # pragma: no cover - robustness
                seed_initvals = None

        try:
            with self.estimation_model:
                pf_idata = pmx_fit(
                    method='pathfinder', num_paths=num_paths,
                    num_draws=num_draws, random_seed=seed,
                    initvals=seed_initvals, **pathfinder_kwargs)
        except Exception as e:  # pragma: no cover - robustness
            return _fallback(f"pathfinder failed ({e})")

        try:
            return self._pathfinder_idata_to_initvals(pf_idata, chains, seed)
        except Exception as e:  # pragma: no cover - robustness
            return _fallback(f"could not convert pathfinder draws ({e})")

    def _pathfinder_idata_to_initvals(self, pf_idata, chains, seed):
        """Convert a Pathfinder ``InferenceData`` into ``chains`` value-variable
        (unconstrained-scale) initval dicts, matching ``initial_point()`` keys.

        Pathfinder returns draws on the **constrained / natural** scale, keyed
        by free-RV name. The sampler wants ``initvals`` on the **value-variable
        (unconstrained) scale** keyed by value-var name (what
        ``model.initial_point()`` uses). For each free RV we apply its
        transform's ``forward`` (e.g. log / logodds) to map the constrained
        draw to the unconstrained value the sampler expects; untransformed RVs
        pass through unchanged.
        """
        import pytensor

        rng = np.random.default_rng(seed)
        post = pf_idata.posterior
        # Flatten (chain, draw) into a single sample axis and pick `chains`
        # distinct draws to disperse the NUTS chains across the typical set.
        n_pf = int(post.sizes['chain'] * post.sizes['draw'])
        stacked = post.stack(_sample=('chain', 'draw'))
        n_take = min(chains, n_pf)
        sel = rng.choice(n_pf, size=n_take, replace=(n_pf < chains))

        model = self.estimation_model
        ip = model.initial_point()  # value-var-name -> array (template)

        # Map each free RV to (value-var name, forward-transform or None),
        # precompiling a forward function per transformed RV for speed.
        rv_specs = []
        for rv, value_var in model.rvs_to_values.items():
            if rv.name not in post:
                # Deterministics / RVs Pathfinder didn't return — skip; the
                # sampler will initialise these from the model default.
                continue
            transform = model.rvs_to_transforms.get(rv, None)
            if transform is None:
                fwd_fn = None
            else:
                inp = pt.tensor(name='_c', dtype='float64',
                                shape=(None,) * rv.ndim)
                fwd_expr = transform.forward(inp, *rv.owner.inputs)
                fwd_fn = pytensor.function(
                    [inp], fwd_expr, on_unused_input='ignore')
            rv_specs.append((rv.name, value_var.name, fwd_fn))

        points = []
        for s in sel:
            pt_dict = {}
            for rv_name, val_name, fwd_fn in rv_specs:
                constrained = np.asarray(
                    stacked[rv_name].isel(_sample=int(s)).values,
                    dtype='float64')
                if fwd_fn is None:
                    unconstrained = constrained
                else:
                    unconstrained = np.asarray(fwd_fn(constrained))
                pt_dict[val_name] = unconstrained.astype(
                    ip[val_name].dtype).reshape(ip[val_name].shape)
            points.append(pt_dict)

        # If Pathfinder produced fewer draws than chains (degenerate), recycle.
        while len(points) < chains:
            points.append({k: v.copy() for k, v in points[-1].items()})
        return points

    def fit_map(self, filter_pars=True, progressbar=True, **kwargs):
        with self.estimation_model:
            pars = pm.find_MAP(progressbar=progressbar, **kwargs)

        if filter_pars:
            pars = {key: pars[key] for key in self.free_parameters}

        if 'subject' in self.estimation_model.coords:
            pars = pd.DataFrame(pars, index=pd.Index(self.estimation_model.coords['subject'], name='subject'))
            pars.columns.name = 'parameter'

        return pars

    def fit_map_individual(self, data=None, flat_prior=True, **kwargs):
        """Fit MLE/MAP estimates for each subject independently (no pooling).

        Loops over subjects, builds a non-hierarchical model on each
        subject's data alone, and returns a DataFrame of point estimates
        in natural (transformed) scale.

        Parameters
        ----------
        data : pd.DataFrame or None
            Trial-level data with a 'subject' index level.  If None, uses
            ``self.paradigm``.
        flat_prior : bool
            If True (default), uses a very wide prior (sigma=100), making
            this effectively maximum-likelihood estimation.  If False, uses
            the model's default prior.
        **kwargs
            Forwarded to ``pm.find_MAP``.

        Returns
        -------
        pd.DataFrame
            Index = subject, columns = free parameter names (transformed scale).
        """
        if data is None:
            data = self.paradigm

        assert 'subject' in data.index.names, \
            "fit_map_individual requires a 'subject' index level."

        subjects = data.index.get_level_values('subject').unique()
        rows = []

        for subj in subjects:
            subj_data = data.xs(subj, level='subject')
            subj_model = self.__class__.__new__(self.__class__)
            subj_model.__dict__.update(self.__dict__)
            subj_model.paradigm = subj_data
            subj_model.build_estimation_model(data=subj_data, hierarchical=False,
                                              flat_prior=flat_prior)
            pars = subj_model.fit_map(filter_pars=False, progressbar=False, **kwargs)
            row = {'subject': subj}
            for param in self.free_parameters:
                row[param] = float(pars[param])
            rows.append(row)

        return pd.DataFrame(rows).set_index('subject')

    def ppc(self, paradigm, idata, n_posterior_samples=200,
            out_of_sample=False, random_seed=None, progressbar=True):
        """Posterior-predictive choices for ``paradigm``.

        Returns
        -------
        pd.DataFrame
            Index: ``paradigm.index`` levels + ``ppc_sample``.
            Single column ``simulated_choice`` (bool).
            Format matches ``DDMMixin.ppc`` / ``RaceMixin.ppc`` so all bauer
            models can be fed into the same downstream summarizers.
        """
        if out_of_sample:
            assert hasattr(self, 'paradigm'), (
                'Model needs paradigm attribute for out-of-sample prediction.')
            if ('subject' in paradigm.index.names) or ('subject' in paradigm.columns):
                coords = {'subject': self.paradigm.index.unique(level='subject')}
                hierarchical = True
                subject_ids = paradigm.index.get_level_values('subject').unique()
                subject_id_to_ix = {s: i for i, s in enumerate(subject_ids)}
            else:
                coords, hierarchical, subject_id_to_ix = None, False, None
            with pm.Model(coords=coords) as self.out_of_sample_model:
                paradigm_ = self._get_paradigm(paradigm=paradigm,
                                               subject_mapping=subject_id_to_ix)
                self.set_paradigm(paradigm_)
                self.build_priors(hierarchical=hierarchical)
                parameters = self.get_parameter_values()
                self.build_likelihood(parameters, save_p_choice=False)
                pp = pm.sample_posterior_predictive(
                    idata, var_names=['ll_bernoulli'],
                    progressbar=progressbar, random_seed=random_seed)
        else:
            with self.estimation_model:
                pp = pm.sample_posterior_predictive(
                    idata, var_names=['ll_bernoulli'],
                    progressbar=progressbar, random_seed=random_seed)

        # arr shape: (chain, draw, n_trials)
        arr = pp['posterior_predictive']['ll_bernoulli'].values.astype(bool)
        n_chain, n_draw, n_trials = arr.shape
        assert n_trials == len(paradigm), (
            f"PPC trial count mismatch: arr has {n_trials}, paradigm has {len(paradigm)}.")

        rng = np.random.default_rng(random_seed)
        n_total = n_chain * n_draw
        n_keep = min(n_posterior_samples, n_total)
        flat = arr.reshape(n_total, n_trials)
        sel = rng.choice(n_total, size=n_keep, replace=False)
        sub = flat[sel].T  # (n_trials, n_keep)

        out = pd.DataFrame(
            sub,
            index=paradigm.index,
            columns=pd.Index(np.arange(n_keep), name='ppc_sample'),
        ).stack().rename('simulated_choice').to_frame()
        out['simulated_choice'] = out['simulated_choice'].astype(bool)
        return out

    def get_trialwise_variable(self, key):

        model = pm.Model.get_context()

        n_trials = pt.cast(model['_data_n'], int)

        if key == 'n1_evidence_mu':
            return model['log(n1)']

        if key == 'n2_evidence_mu':
            return model['log(n2)']

        if model[f'{key}'].ndim == 1:
            return model[f'{key}'][model['subject_ix']]
        elif model[f'{key}'].ndim == 0:
            return pt.tile(model[f'{key}'], n_trials)

    def build_hierarchical_nodes(self, name, mu_intercept=None, sigma_intercept=None,
                                 cauchy_sigma_intercept=None, transform='identity',
                                 min_value=0.0, beta_mu_mean=0.05, beta_mu_kappa=8.0,
                                 beta_kappa_sd=3.0, **kwargs):
        """Build a hierarchical (group_mu, group_sd, per-subject offset) node.

        ``transform`` ∈ {'identity', 'softplus', 'logistic', 'beta'}. When
        ``transform`` is 'softplus' and ``min_value > 0``, the transformed
        parameter has a hard lower bound: ``param = min_value + softplus(x)``.
        Used to keep e.g. the DDM/RDM threshold ``a`` away from the a→0 collapse
        mode.

        Hierarchical-Beta group prior (``transform='beta'``)
        ----------------------------------------------------
        A per-subject rate in ``(0, 1)`` whose population density is
        concentrated near 0 with a heavy upper tail — the "1/x"-like shape — for
        parameters such as a lapse / outlier rate, where most subjects are ≈ 0
        but a minority genuinely lapse a lot. This replaces the logit-Normal
        parameterization (``transform='logistic'``), which **funnels** when the
        true rate ≈ 0: the per-subject logit → −∞, the group SD inflates without
        bound, and NUTS diverges.

        The model is::

            mu     ~ Beta(beta_mu_mean·beta_mu_kappa, (1-beta_mu_mean)·beta_mu_kappa)
            kappa  = 2 + exp(log_kappa),   log_kappa ~ Normal(0, beta_kappa_sd)
            p[s]   ~ Beta(alpha, beta),    alpha = mu·kappa, beta = (1-mu)·kappa

        ``mu`` is the group-mean rate (prior mean ``beta_mu_mean``, default
        0.05); ``kappa`` is the population concentration. The per-subject density
        is ``∝ p^(alpha-1)(1-p)^(beta-1)``: when ``alpha = mu·kappa < 1`` (small
        ``mu``) this is an integrable spike at 0 (the "1/x" shape) with a fat
        upper tail, exactly matching "most subjects ≈ 0, a few disengaged". The
        ``kappa = 2 + exp(...)`` floor keeps ``beta = (1-mu)·kappa > 1`` so the
        density is bounded near 1 (no spurious upper-edge spike) while still
        allowing ``alpha < 1``. ``p[s]`` is sampled directly as a ``pm.Beta``,
        whose default log-odds transform gives NUTS a clean unconstrained
        geometry without the logit funnel (the per-subject scale is set by the
        *data-informed* Beta, not a shared inflating SD). Group nodes
        ``{name}_mu`` and ``{name}_kappa`` are exposed for reporting (so
        ``get_groupwise_parameter_estimates`` keeps working via ``{name}_mu``).
        """

        if transform == 'beta':
            a0 = beta_mu_mean * beta_mu_kappa
            b0 = (1.0 - beta_mu_mean) * beta_mu_kappa
            group_mu = pm.Beta(f'{name}_mu', alpha=a0, beta=b0)
            log_kappa = pm.Normal(f'{name}_log_kappa', mu=0.0, sigma=beta_kappa_sd)
            group_kappa = pm.Deterministic(f'{name}_kappa',
                                           2.0 + pt.exp(log_kappa))
            alpha = group_mu * group_kappa
            beta = (1.0 - group_mu) * group_kappa
            return pm.Beta(name, alpha=alpha, beta=beta, dims=('subject',))

        if mu_intercept is None:
            mu_intercept = 0.0

        if sigma_intercept is None:
            sigma_intercept = .5

        if cauchy_sigma_intercept is None:
            cauchy_sigma_intercept = 0.25

        def _softplus_with_floor(x):
            if min_value:
                return min_value + pt.softplus(x)
            return pt.softplus(x)

        if transform == 'identity':
            group_mu = pm.Normal(f"{name}_mu",
                                 mu=mu_intercept,
                                 sigma=sigma_intercept)
        elif transform == 'softplus':
            group_mu = pm.Normal(f"{name}_mu_untransformed",
                                 mu=mu_intercept,
                                 sigma=sigma_intercept)

            pm.Deterministic(name=f'{name}_mu', var=_softplus_with_floor(group_mu))
        elif transform == 'logistic':
            group_mu = pm.Normal(f"{name}_mu_untransformed",
                                 mu=mu_intercept,
                                 sigma=sigma_intercept)

            pm.Deterministic(name=f'{name}_mu', var=logistic(group_mu))
        else:
            raise NotImplementedError

        group_sd = _group_sd_rv(f'{name}_sd', cauchy_sigma_intercept,
                                getattr(self, 'group_sd_dist', 'halfcauchy'))
        subject_offset = pm.Normal(f'{name}_offset', mu=0, sigma=1, dims=('subject',))

        if transform == 'identity':
            return pm.Deterministic(f'{name}', group_mu + group_sd * subject_offset, dims=('subject',))
        else:
            subjectwise_untrans = pm.Deterministic(f'{name}_untransformed', group_mu + group_sd * subject_offset, dims=('subject',))

            if transform == 'softplus':
                return pm.Deterministic(name=name, var=_softplus_with_floor(subjectwise_untrans), dims=('subject',))
            elif transform == 'logistic':
                return pm.Deterministic(name=name, var=logistic(subjectwise_untrans), dims=('subject',))
            else:
                raise Exception

    def get_subjectwise_parameter_estimates(self, idata=None, parameters=None):

        idata = self._get_idata(idata)
        parameters = list(self._get_parameters(parameters).keys())

        pars = idata.posterior[parameters].to_dataframe()
        pars.columns.name = 'parameter'

        return pars

    def get_groupwise_parameter_estimates(self, idata=None, parameters=None, include_sd=True):

        idata = self._get_idata(idata)
        parameters = self._get_parameters(parameters)

        mu_pars = pd.concat([idata.posterior[par + '_mu'].to_dataframe() for par in self.free_parameters], axis=1, keys=parameters, names=['parameter']).droplevel(1, axis=1)

        if include_sd:
            # 'beta' group priors expose a concentration ({par}_kappa) instead
            # of a Normal scale ({par}_sd); fall back to it so reporting works.
            def _spread(par):
                if par + '_sd' in idata.posterior:
                    return idata.posterior[par + '_sd'].to_dataframe()
                return idata.posterior[par + '_kappa'].to_dataframe()
            sd_pars = pd.concat([_spread(par) for par in self.free_parameters], axis=1, keys=self.free_parameters, names=['parameter']).droplevel(1, axis=1)
            pars = pd.concat((mu_pars, sd_pars), keys=['mu', 'sd'], names=['type'], axis=1)

        else:
            pars = pd.concat((mu_pars,), keys=['mu'], names=['type'], axis=1)

        pars.columns.name = 'parameter'

        return pars

    def _get_idata(self, idata):
        if idata is None:
            idata = self.idata
            if idata is None:
                raise ValueError('No idata found. Please run sample() first.')

        return idata

    def _get_parameters(self, parameters):
        if parameters is None:
            parameters = self.free_parameters

        return parameters

    def sample_parameters_from_prior(self, n_subjects=None):
        samples_pars = {}
        for key, pars in self.free_parameters.items():

            if 'sigma_intercept' not in pars.keys():
                pars['sigma_intercept'] = 1.
            if 'transform' not in pars.keys():
                pars['transform'] = 'identity'

            if pars['transform'] == 'beta':
                mean = pars.get('beta_mu_mean', 0.05)
                kappa = pars.get('beta_mu_kappa', 8.0)
                samples_pars[key] = np.random.beta(mean * kappa,
                                                   (1.0 - mean) * kappa,
                                                   n_subjects)
                continue

            samples_pars[key] = np.random.normal(pars['mu_intercept'], pars['sigma_intercept'], n_subjects)
            if pars['transform'] == 'softplus':
                samples_pars[key] = softplus_np(samples_pars[key])
            elif pars['transform'] == 'logistic':
                samples_pars[key] = logistic_np(samples_pars[key])

        if n_subjects:
            sample_pars = pd.DataFrame(samples_pars, index=pd.Index(np.arange(1, n_subjects + 1), name='subject'))
            sample_pars.columns.name = 'parameter'
            return sample_pars
        else:
            return samples_pars

    def forward_transform(self, data, parameter):
        transform = self.free_parameters[parameter]['transform']

        if transform in ('identity', 'beta'):
            # 'beta' rates already live in (0, 1) — natural == sampling scale.
            return data
        elif transform == 'softplus':
            return softplus_np(data)
        elif transform == 'logistic':
            return logistic_np(data)

    def backward_transform(self, data, parameter):
        transform = self.free_parameters[parameter]['transform']

        if transform in ('identity', 'beta'):
            return data
        elif transform == 'softplus':
            return inverse_softplus_np(data)
        elif transform == 'logistic':
            return logit_np(data)

    def get_example_paradigm(self, n_subjects=None):
        if n_subjects is None:
            return self._get_example_paradigm()
        else:
            return pd.concat([self._get_example_paradigm() for _ in range(n_subjects)], keys=np.arange(1, n_subjects + 1), names=['subject'])

    def get_model_inputs(self, parameters):

        model = pm.Model.get_context()

        model_inputs = {}

        for key in self.base_parameters:
            model_inputs[key] = parameters[key]

        for key in self.paradigm_keys:
            model_inputs[key] = model[key]

        return model_inputs


# BaseModel can't translate its own __init__ via __init_subclass__ (that hook
# only fires for subclasses), so wrap it explicitly here.
BaseModel.__init__ = _translate_deprecated_kwargs(BaseModel.__init__)


def _lapse_param_spec(group, mu_mean, mu_kappa, kappa_sd):
    """Free-parameter spec dict for a lapse / outlier rate in (0, 1).

    ``group='logit_normal'`` (default) reproduces the historical logit-Normal
    hierarchical prior (centred at ``mu_mean``). ``group='beta'`` selects the
    hierarchical-Beta "1/x"-like group prior (see
    :meth:`BaseModel.build_hierarchical_nodes`), which is concentrated near 0
    with a heavy upper tail and removes the logit funnel that arises when the
    true rate ≈ 0.
    """
    if group == 'beta':
        return {'transform': 'beta',
                'beta_mu_mean': mu_mean,
                'beta_mu_kappa': mu_kappa,
                'beta_kappa_sd': kappa_sd}
    elif group == 'logit_normal':
        return {'mu_intercept': logit_np(mu_mean), 'transform': 'logistic'}
    raise ValueError(
        f"lapse_group must be 'logit_normal' or 'beta', got {group!r}.")


class LapseModel(BaseModel):
    """Static-choice model with a per-subject random-lapse rate ``p_lapse``.

    ``lapse_group`` selects the group prior on the per-subject lapse rate:
    ``'logit_normal'`` (legacy opt-in; Beta is now the default) or ``'beta'`` (the
    heavy-tailed "1/x"-like hierarchical Beta; see
    :meth:`BaseModel.build_hierarchical_nodes`). The Beta option is preferable
    when most subjects lapse ≈ 0 (it avoids the logit funnel).
    """

    lapse_group = 'beta'
    lapse_mu_mean = 0.02
    lapse_mu_kappa = 8.0
    lapse_kappa_sd = 3.0

    def get_free_parameters(self):
        pars = super().get_free_parameters()

        pars['p_lapse'] = _lapse_param_spec(self.lapse_group,
                                            self.lapse_mu_mean,
                                            self.lapse_mu_kappa,
                                            self.lapse_kappa_sd)
        return pars

    def _get_choice_predictions(self, model_inputs):
        p_choice = super()._get_choice_predictions(model_inputs)
        p_choice = p_choice * (1 - model_inputs['p_lapse']) + 0.5 * model_inputs['p_lapse']
        return p_choice

    def get_model_inputs(self, parameters):
        model_inputs = super().get_model_inputs(parameters)
        model_inputs['p_lapse'] = parameters['p_lapse']
        return model_inputs


class RTLapseMixin(object):
    """RT-aware lapse / outlier-contaminant mixture for sequential-sampling
    (DDM / RDM) likelihoods, mirroring HSSM's ``p_outlier`` mechanism.

    The static-choice :class:`LapseModel` lapses *choices* only
    (``p = p·(1-λ) + 0.5·λ``); that is correct for the cumulative-normal
    Bernoulli likelihood but **wrong** for a likelihood defined jointly over
    ``(rt, choice)``. A response-time model needs an RT-aware contaminant: on a
    fraction ``p_outlier`` of trials the observation is assumed to come from a
    process unrelated to the decision (fast guesses, lapses of attention,
    delayed/anticipatory responses), modelled by a flat density over RT.

    Following HSSM (``hssm.distribution_utils.dist``), the per-trial
    log-likelihood becomes the numerically-stable mixture::

        logp = log( (1 - p_outlier) * exp(ll_model)
                    + p_outlier * exp(lapse_logp) + 1e-29 )

    where ``ll_model`` is the WFPT (DDM) / Wald-race (RDM) log-density and
    ``lapse_logp`` is the contaminant log-density.

    Contaminant density
    -------------------
    HSSM's default lapse distribution is ``Uniform(0, lapse_upper)`` over RT
    (``lapse_upper`` = 20 s), and the contaminant choice is 50/50. The joint
    (rt, choice) contaminant density is therefore::

        f_lapse(rt, choice) = 0.5 / lapse_upper      for 0 <= rt <= lapse_upper
                            = 0                       otherwise

    so ``lapse_logp = log(0.5) - log(lapse_upper)`` — a constant, independent of
    the trial (every observed RT in a fitted dataset is well within the bound).
    This 0.5/upper joint form (RT *and* choice uniform) is the conceptually
    complete contaminant; HSSM's code path happens to score only the RT margin
    (``-log(lapse_upper)``), but since the term is an additive constant inside
    the mixture this changes only the absolute scale of the contaminant weight,
    not the inference about decision parameters. The factor is documented here
    for reproducibility. Set ``lapse_choice_5050=False`` to drop the ``log(0.5)``
    and exactly match HSSM's RT-only convention.

    Parameters
    ----------
    lapse_upper : float
        Upper bound (seconds) of the uniform contaminant RT density. Default
        20.0 (HSSM default). Must exceed the largest observed RT.
    lapse_choice_5050 : bool
        If True (default), the contaminant assigns 50/50 over the two
        responses (joint density 0.5/upper). If False, match HSSM's RT-only
        margin (1/upper).

    The free parameter is named ``p_outlier`` (HSSM's name).

    Group prior on ``p_outlier``
    ----------------------------
    ``lapse_group`` selects how the *per-subject* outlier rate is regularized
    across subjects:

    - ``'logit_normal'`` (legacy opt-in; Beta is now the default): hierarchical
      logit-Normal, centred at ``logit(0.05)``. This **funnels** when subjects'
      true lapse ≈ 0 — the per-subject logit → −∞, the group SD inflates, and
      NUTS diverges.
    - ``'beta'``: hierarchical Beta "1/x"-like group prior (group mean μ +
      concentration κ; density ``∝ p^(μκ-1)`` — a spike at 0 with a heavy upper
      tail). See :meth:`BaseModel.build_hierarchical_nodes`. Recommended on
      clean data, where it removes the funnel because the per-subject rate lives
      in (0, 1) directly rather than on an unbounded logit.
    """

    # HSSM-matching defaults; overridable per-instance via __init__ kwargs of
    # the concrete classes, which set these attributes before super().__init__.
    lapse_upper = 20.0
    lapse_choice_5050 = True
    # ``p_outlier`` mirrors HSSM's API:
    #   - a FLOAT (default 0.05, HSSM's default) -> the contaminant rate is
    #     FIXED at that value; it is NOT a sampled parameter, so it cannot
    #     funnel/diverge. This is the recommended/default mode.
    #   - 'hierarchical' -> per-subject estimated rate (bauer extension; weakly
    #     identified in the WFPT/Wald mixture, prone to divergences -- opt-in).
    # Estimating a single (non-hierarchical) rate is not yet supported; HSSM's
    # own default is the fixed value anyway.
    p_outlier = 0.05
    # Whether simulate()/ppc() GENERATE contaminant trials.
    # NOTE on HSSM: HSSM's posterior predictive *does* generate them — its
    # RandomVariable.rng_fn applies the lapse (`_apply_lapse_model`) after the
    # clean ssms draw; only its standalone `simulate_data` helper is clean.
    # We nonetheless default to False on the (defensible, arguably cleaner)
    # interpretation that p_outlier is a *likelihood robustness* device — a way
    # to down-weight untrusted trials — NOT a generative process. So the
    # predictor reflects the clean decision model; a PPC that "misses" the RT
    # tail is then honestly telling you the decision model doesn't explain those
    # trials. Set True to (a) match HSSM's posterior predictive exactly and
    # (b) avoid that (expected) tail "misfit" when overlaying a PPC on real data
    # (a fraction p_outlier become Uniform(0, lapse_upper) x 50/50 contaminants).
    simulate_contaminant = False
    # Group prior for the per-subject rate (only used when p_outlier='hierarchical').
    lapse_group = 'beta'
    lapse_mu_mean = 0.05
    lapse_mu_kappa = 8.0
    lapse_kappa_sd = 3.0

    def get_free_parameters(self):
        pars = super().get_free_parameters()
        if isinstance(self.p_outlier, (int, float)):
            # Fixed contaminant rate (HSSM default): not a free parameter.
            return pars
        spec = _lapse_param_spec(self.lapse_group,
                                 self.lapse_mu_mean,
                                 self.lapse_mu_kappa,
                                 self.lapse_kappa_sd)
        if self.lapse_group == 'logit_normal':
            # Preserve the historical sigma_intercept=0.5 on the logit prior.
            spec['sigma_intercept'] = 0.5
        pars['p_outlier'] = spec
        return pars

    def get_model_inputs(self, parameters):
        model_inputs = super().get_model_inputs(parameters)
        if isinstance(self.p_outlier, (int, float)):
            model_inputs['p_outlier'] = pt.constant(float(self.p_outlier))
        else:
            model_inputs['p_outlier'] = parameters['p_outlier']
        return model_inputs

    def _lapse_logp_const(self):
        """Constant log-density of the uniform RT (× 50/50 choice) contaminant."""
        const = -np.log(self.lapse_upper)
        if self.lapse_choice_5050:
            const += np.log(0.5)
        return float(const)

    def _mix_with_lapse(self, ll_model, p_outlier):
        """HSSM-style numerically-stable lapse mixture of per-trial log-densities.

        ``ll_model`` is the (per-trial) WFPT / Wald-race log-likelihood; the
        contaminant is a flat density with constant log-value
        :meth:`_lapse_logp_const`. Returns the mixed per-trial log-likelihood.
        """
        lapse_logp = pt.constant(self._lapse_logp_const())
        # log( (1-p)·exp(ll) + p·exp(lapse) + 1e-29 ), via logaddexp for stability.
        log_keep = pt.log1p(-p_outlier) + ll_model
        log_lapse = pt.log(p_outlier) + lapse_logp
        mixed = pt.logaddexp(log_keep, log_lapse)
        # Match HSSM's +1e-29 guard against log(0) when both terms underflow.
        return pt.logaddexp(mixed, pt.constant(np.log(1e-29)))


class RegressionModel(BaseModel):

    def __init__(self, regressors=None):

        if regressors is None:
            self.regressors = {}
        else:
            self.regressors = regressors

        self.design_matrices = {}

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        paradigm_ = super()._get_paradigm(paradigm, subject_mapping=subject_mapping)

        free_parameters = self.get_free_parameters()

        for key in free_parameters:
            dm = self.build_design_matrix(paradigm, key)
            self.design_matrices[key] = dm
            paradigm_[f'_dm_{key}'] = np.asarray(dm)

        return paradigm_

    def build_design_matrix(self, data, parameter):
        if parameter not in self.regressors:
            self.regressors[parameter] = '1'

        return dmatrix(self.regressors[parameter], data)

    def rebuild_design_matrix(self, paradigm, parameter):
        assert (hasattr(self, 'design_matrices')), 'Model needs to have design matrices as an attribute (model.design_matrices...) based on original data for rebuilding design matrices (HINT: use `.build_estimation_model()`)'

        design_info = self.design_matrices[parameter].design_info

        return build_design_matrices([design_info], paradigm)[0]

    def get_trialwise_variable(self, key):

        model = pm.Model.get_context()

        if 'transform' in self.free_parameters[key]:
            transform = self.free_parameters[key]['transform']
        else:
            transform = 'identity'

        dm = model[f'_dm_{key}']

        if key in ['n1_evidence_mu', 'n2_evidence_mu']:
            if key in self.design_matrices.keys():
                trialwise_pars = model[f'log({key[:2]})'] + pt.sum(model[key][model['subject_ix']] * dm, 1)
            else:
                if key == 'n1_evidence_mu':
                    return model['log(n1)']

                if key == 'n2_evidence_mu':
                    return model['log(n2)']

        else:
            pars = model[key]
            if pars.ndim == 1:
                trialwise_pars = pt.sum(pars * dm, 1)
            elif model[key].ndim == 2:
                trialwise_pars = pt.sum(pars[model['subject_ix']] * dm, 1)
            else:
                raise ValueError(f'Unknown dimensionality of {key}')

            if transform == 'softplus':
                trialwise_pars = pt.softplus(trialwise_pars)
                # Apply the same min_value floor BaseModel.build_hierarchical_nodes
                # applies to non-regressed softplus parameters (e.g. DDM `a`, `t0`).
                # Without this, regressed versions of those parameters can collapse
                # toward 0 even though the prior intercept assumes a hard lower bound.
                min_value = self.free_parameters[key].get('min_value', 0.0)
                if min_value:
                    trialwise_pars = min_value + trialwise_pars
            elif transform == 'logistic':
                trialwise_pars = logistic(trialwise_pars)

        return trialwise_pars

    def build_prior(self, name, mu_intercept=None, sigma_intercept=None,
                    sigma_regressors=1., transform='identity',
                    flat_prior=False, **_unused):
        # ``flat_prior`` is accepted (and currently ignored) so the
        # non-hierarchical dispatch (``build_priors(hierarchical=False)``)
        # works for regression models too; the legacy hierarchical path
        # already supports it.
        if sigma_intercept is None:
            sigma_intercept = 1.0
        model = pm.Model.get_context()
        model.add_coord(f'{name}_regressors', self.design_matrices[name].design_info.column_names)

        mu = np.zeros(self.design_matrices[name].shape[1])
        sigma = np.ones(self.design_matrices[name].shape[1]) * sigma_regressors

        if self.design_matrices[name].design_info.column_names[0] == 'Intercept':

            if name in ['n1_evidence_mu', 'n2_evidence_mu']:
                warnings.warn(f'{name} has an Intercept-regressors. This is most likely NOT a good idea.')

            # The regression happens on the *untransformed* scale (the
            # transform — softplus / logistic / identity — is applied later
            # in get_trialwise_variable after the design-matrix product), so
            # the Intercept prior is always Normal(mu_intercept,
            # sigma_intercept) regardless of which transform the parameter
            # uses. Previously the softplus branch was missing, silently
            # defaulting softplus parameters to Normal(0, sigma_regressors)
            # — which wrecked NUTS mixing on every regression DDM fit.
            mu[0] = mu_intercept
            sigma[0] = sigma_intercept

        pm.Normal(name, mu=mu, sigma=sigma, dims=(f'{name}_regressors',))

    def build_hierarchical_nodes(self, name, mu_intercept=0.0, sigma_intercept=None,
                                 cauchy_sigma_intercept=None, sigma_regressors=1.,
                                 cauchy_sigma_regressors=0.25, transform='identity',
                                 min_value=0.0, **kwargs):
        # `min_value` and the transform are applied in get_trialwise_variable
        # (which sees the design-matrix product); accepted here so that
        # parameters carrying these kwargs (e.g. DDM `a`, `t0`) don't crash the
        # regression-model dispatch.
        # sigma_intercept / cauchy_sigma_intercept default to None (resolved
        # to 0.5 / 0.25 below) to match BaseModel.build_hierarchical_nodes,
        # so that intercept-only regression models have the same Intercept
        # prior as the equivalent basic model.
        if sigma_intercept is None:
            sigma_intercept = 0.5
        if cauchy_sigma_intercept is None:
            cauchy_sigma_intercept = 0.25

        model = pm.Model.get_context()
        model.add_coord(f'{name}_regressors', self.design_matrices[name].design_info.column_names)

        mu = np.zeros(self.design_matrices[name].shape[1])
        sigma = np.ones(self.design_matrices[name].shape[1]) * sigma_regressors
        cauchy_sigma = np.ones(self.design_matrices[name].shape[1]) * cauchy_sigma_regressors
        cauchy_sigma[0] = cauchy_sigma_intercept

        if self.design_matrices[name].design_info.column_names[0] == 'Intercept':

            if name in ['n1_evidence_mu', 'n2_evidence_mu']:
                warnings.warn(f'{name} has an Intercept-regressors. This is most likely NOT a good idea.')

            # See note in build_prior: the regression operates on the
            # *untransformed* scale, so the Intercept prior is the same
            # Normal(mu_intercept, sigma_intercept) for all three
            # transforms. The previously-missing softplus branch was the
            # source of the regression-DDM convergence pathology.
            mu[0] = mu_intercept
            sigma[0] = sigma_intercept

        group_mu = pm.Normal(f"{name}_mu",
                             mu=mu,
                             sigma=sigma,
                             dims=(f'{name}_regressors',))

        group_sd = _group_sd_rv(f'{name}_sd', cauchy_sigma,
                                getattr(self, 'group_sd_dist', 'halfcauchy'),
                                dims=(f'{name}_regressors',))
        subject_offset = pm.Normal(f'{name}_offset', mu=0, sigma=1, dims=('subject', f'{name}_regressors'))

        return pm.Deterministic(name, group_mu + group_sd * subject_offset, dims=('subject', f'{name}_regressors'))

    def fit_map(self, filter_pars=True, **kwargs):

        def convert_map(map_parameters):
            df = []

            hierarchical = 'subject' in self.estimation_model.coords

            if hierarchical:
                subjects = self.estimation_model.coords['subject']

            parameters = self.design_matrices.keys()
            for parameter, dm in self.design_matrices.items():
                if hierarchical:
                    df.append(pd.DataFrame(map_parameters[parameter],
                                           index=pd.Index(subjects, name='subject'), columns=pd.Index(dm.design_info.column_names, name='dm_key')))
                else:
                    df.append(pd.Series(map_parameters[parameter],
                                        index=pd.Index(dm.design_info.column_names, name='dm_key')))

            df = pd.concat(df, keys=parameters, axis=1)

            if not hierarchical:
                df = df.stack().to_frame().T.swaplevel(0, 1, axis=1).sort_index(axis=1)

            df.columns.names = ['parameter', 'dm_key']

            return df

        with self.estimation_model:
            pars = pm.find_MAP(**kwargs)

        return convert_map(pars)

    def build_prediction_model(self, paradigm, parameters,):

        if isinstance(parameters, pd.DataFrame):
            hierarchical = True
        elif isinstance(parameters, dict):
            hierarchical = False
        else:
            raise ValueError("Parameters should be a dictionary or a DataFrame.")

        with pm.Model() as self.prediction_model:
            paradigm = self._get_paradigm(paradigm=paradigm)
            self.set_paradigm(paradigm)

            if hierarchical:
                n_subjects = np.unique(self.prediction_model['subject_ix'].eval()).shape[0]
                assert (len(parameters) == n_subjects), f'Number of subjects in data ({n_subjects}) does not match number of subjects in parameters ({len(parameters)})'

            for key in self.free_parameters.keys():
                dm_keys = self.design_matrices[key].design_info.column_names
                if hierarchical:
                    assert parameters[key].columns.tolist() == dm_keys, f'Parameter {key} needs the following columns: {dm_keys} (in that order)'
                    pm.Data(key, parameters[key])
                else:
                    value = [parameters[(key, dm_key)] for dm_key in dm_keys]
                    pm.Data(key, value)

            parameters = self.get_parameter_values()
            model_inputs = self.get_model_inputs(parameters)

            p = pm.Deterministic('p', var=self._get_choice_predictions(model_inputs))
            pm.Bernoulli('ll_bernoulli', p=p)

    def sample_parameters_from_prior(self, paradigm, n_subjects=None):

        samples_pars = {}

        for par_key, pars in self.free_parameters.items():
            dm = self.build_design_matrix(paradigm, par_key)

            dm_keys = dm.design_info.column_names

            for dm_key in dm_keys:
                key = (par_key, dm_key)

                if dm_keys == 'Intercept':
                    if 'sigma_intercept' not in pars.keys():
                        pars['sigma_intercept'] = 1.
                    samples_pars[key] = np.random.normal(pars['mu_intercept'], pars['sigma_intercept'], n_subjects)

                else:
                    samples_pars[key] = np.random.normal(0, 1, n_subjects)

        if n_subjects:
            return pd.DataFrame(samples_pars, index=pd.Index(np.arange(1, n_subjects + 1), name='subject'))
        else:
            return samples_pars

    def get_conditionwise_parameters(self, idata, conditions, group=True):

        parameter_values = []
        free_parameters = self.get_free_parameters()

        assert (hasattr(self, 'paradigm')), 'Model needs to have original paradigm as an attribute (model.paradigm...) for calcuating conditionwise parameters'

        for parameter in free_parameters.keys():
            dm = self.rebuild_design_matrix(conditions, parameter)

            # Hierarchical regression fits store the group-mean coefficients as
            # `{parameter}_mu`; per-subject (non-hierarchical) fits only have
            # `{parameter}`. Prefer the group-mean when asked for the group and
            # it exists, otherwise fall back to the per-subject coefficients.
            if group and f'{parameter}_mu' in idata.posterior:
                trace = idata.posterior[f'{parameter}_mu'].to_dataframe().unstack(-1)
            else:
                trace = idata.posterior[f'{parameter}'].to_dataframe().unstack(-1)

            par = (np.asarray(trace).reshape((1, -1, dm.shape[1])) @ dm.T)[0]

            if conditions.shape[1] == 1:
                par = pd.DataFrame(par, columns=pd.Index(conditions.iloc[:, 0]), index=trace.index)
            else:
                par = pd.DataFrame(par, columns=pd.MultiIndex.from_frame(conditions), index=trace.index)

            if free_parameters[parameter]['transform'] == 'logistic':
                par = logistic_np(par)
            elif free_parameters[parameter]['transform'] == 'softplus':
                par = np.log1p(np.exp(par))

            parameter_values.append(par)

        return pd.concat(parameter_values, axis=0, keys=free_parameters.keys(), names=['parameter'])
