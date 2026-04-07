#!/usr/bin/env python
"""Fit numerosity estimation models via MCMC.

Usage:
    # Single subject (non-hierarchical):
    python fit_numerosity.py 1 --model log_encoding

    # Hierarchical (all subjects):
    python fit_numerosity.py --hierarchical --model log_encoding

    # Hierarchical subset:
    python fit_numerosity.py --hierarchical --subjects 1 2 3 --model log_encoding

Saves arviz InferenceData as netcdf.
"""

import argparse
import os
import numpy as np
import pandas as pd


def get_paradigm(subject_ids=None):
    """Load and prepare paradigm."""
    from bauer.utils.data import load_neuralpriors
    df = load_neuralpriors()

    if subject_ids is not None:
        df = df[df.index.get_level_values('subject').isin(subject_ids)]

    all_paradigms = []
    for sub_id in df.index.get_level_values('subject').unique():
        sub_data = df.xs(sub_id, level='subject').reset_index()
        p = sub_data[['n', 'response', 'range']].dropna(subset=['response']).copy()
        p['n'] = p['n'].astype(float)
        p['subject'] = sub_id
        all_paradigms.append(p)

    paradigm = pd.concat(all_paradigms, ignore_index=True)
    paradigm = paradigm.set_index(['subject', paradigm.groupby('subject').cumcount()])
    paradigm.index.names = ['subject', 'trial']

    subjects = paradigm.index.get_level_values('subject').unique()
    n_trials = len(paradigm)
    n_narrow = (paradigm['range'] == 'narrow').sum()
    n_wide = (paradigm['range'] == 'wide').sum()
    print(f"Loaded {len(subjects)} subjects, {n_trials} trials "
          f"({n_narrow} narrow, {n_wide} wide)")

    return paradigm


def make_model(paradigm, model_name, grid_resolution):
    """Create model instance."""
    if model_name == 'log_encoding':
        from bauer.numerosity import LogEncodingEstimationModel
        return LogEncodingEstimationModel(
            paradigm, grid_resolution=grid_resolution,
            n_min=10, n_max=40, response_bin_width=1.0)

    elif model_name == 'flexible_shared':
        from bauer.numerosity import FlexibleEncodingEstimationModel
        return FlexibleEncodingEstimationModel(
            paradigm, grid_resolution=grid_resolution, n_poly=6,
            n_min=10, n_max=40, response_bin_width=1.0,
            condition_specific_encoding=False)

    elif model_name == 'flexible_condition':
        from bauer.numerosity import FlexibleEncodingEstimationModel
        return FlexibleEncodingEstimationModel(
            paradigm, grid_resolution=grid_resolution, n_poly=6,
            n_min=10, n_max=40, response_bin_width=1.0,
            condition_specific_encoding=True)

    elif model_name == 'efficient_encoding':
        from bauer.numerosity import EfficientEncodingEstimationModel
        return EfficientEncodingEstimationModel(
            paradigm, grid_resolution=grid_resolution,
            n_min=10, n_max=40, response_bin_width=1.0)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', type=int, nargs='?', default=None,
                        help='Subject ID (for single-subject fit)')
    parser.add_argument('--hierarchical', action='store_true',
                        help='Fit hierarchical model across all subjects')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help='Subset of subjects for hierarchical fit')
    parser.add_argument('--model', default='log_encoding',
                        choices=['log_encoding', 'flexible_shared',
                                 'flexible_condition', 'efficient_encoding'])
    parser.add_argument('--grid-resolution', type=int, default=31)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--target-accept', type=float, default=0.95)
    parser.add_argument('--nuts-sampler', default='pymc',
                        choices=['pymc', 'numpyro', 'nutpie'],
                        help='NUTS implementation (numpyro uses JAX/GPU)')
    parser.add_argument('--output-dir', default='results/numerosity')
    args = parser.parse_args()

    if args.hierarchical:
        subject_ids = args.subjects  # None = all subjects
        paradigm = get_paradigm(subject_ids)
        hierarchical = True
        label = 'hierarchical'
        if subject_ids:
            label += f'_n{len(subject_ids)}'
    else:
        if args.subject is None:
            parser.error("Provide a subject ID, or use --hierarchical")
        paradigm = get_paradigm([args.subject])
        hierarchical = False
        label = f'sub-{args.subject:02d}'

    model = make_model(paradigm, args.model, args.grid_resolution)

    print(f"Building model: {args.model} (grid={args.grid_resolution}, "
          f"hierarchical={hierarchical})")
    model.build_estimation_model(paradigm, hierarchical=hierarchical,
                                 flat_prior=not hierarchical,
                                 save_p_choice=False)

    print(f"Free parameters: {list(model.free_parameters.keys())}")
    print(f"Sampling: {args.chains} chains, {args.tune} tune + {args.draws} draws")

    sample_kwargs = dict(draws=args.draws, tune=args.tune,
                         target_accept=args.target_accept,
                         chains=args.chains, cores=1)
    if args.nuts_sampler != 'pymc':
        sample_kwargs['nuts_sampler'] = args.nuts_sampler
    idata = model.sample(**sample_kwargs)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    fn = os.path.join(args.output_dir,
                      f'{label}_model-{args.model}_grid-{args.grid_resolution}.netcdf')
    idata.to_netcdf(fn)
    print(f"Saved to {fn}")

    # Summary
    import arviz as az
    var_names = [k for k in model.free_parameters.keys()
                 if not k.startswith('enc_')]
    if hierarchical:
        var_names = [f'{k}_mu' for k in var_names] + [f'{k}_sd' for k in var_names]
    print("\nPosterior summary:")
    print(az.summary(idata, var_names=var_names))


if __name__ == '__main__':
    main()
