#!/usr/bin/env python
"""Fit numerosity estimation models via MCMC.

Usage:
    python fit_numerosity.py SUBJECT [--model MODEL] [--grid-resolution N]
                             [--draws N] [--tune N] [--chains N]
                             [--output-dir DIR]

Fits the specified model to a single subject's data (both narrow and wide
conditions jointly) using MCMC sampling. Saves arviz InferenceData as netcdf.
"""

import argparse
import os
import numpy as np
import pandas as pd


def get_paradigm(subject_id):
    """Load and prepare paradigm for a single subject."""
    from bauer.utils.data import load_neuralpriors
    df = load_neuralpriors()

    sub_data = df.xs(subject_id, level='subject').reset_index()
    paradigm = sub_data[['n', 'response', 'range']].dropna(subset=['response']).copy()
    paradigm['n'] = paradigm['n'].astype(float)
    paradigm.index = pd.MultiIndex.from_arrays(
        [np.full(len(paradigm), subject_id), range(len(paradigm))],
        names=['subject', 'trial'])

    print(f"Subject {subject_id}: {len(paradigm)} trials "
          f"({(paradigm['range']=='narrow').sum()} narrow, "
          f"{(paradigm['range']=='wide').sum()} wide)")

    return paradigm


def fit_model(paradigm, model_name, grid_resolution, draws, tune, chains,
              target_accept):
    """Build and sample the specified model."""
    import pymc as pm

    if model_name == 'log_encoding':
        from bauer.numerosity import LogEncodingEstimationModel
        model = LogEncodingEstimationModel(
            paradigm, grid_resolution=grid_resolution,
            n_min=10, n_max=40, response_bin_width=1.0)

    elif model_name == 'flexible_shared':
        from bauer.numerosity import FlexibleEncodingEstimationModel
        model = FlexibleEncodingEstimationModel(
            paradigm, grid_resolution=grid_resolution, n_poly=6,
            n_min=10, n_max=40, response_bin_width=1.0,
            condition_specific_encoding=False)

    elif model_name == 'flexible_condition':
        from bauer.numerosity import FlexibleEncodingEstimationModel
        model = FlexibleEncodingEstimationModel(
            paradigm, grid_resolution=grid_resolution, n_poly=6,
            n_min=10, n_max=40, response_bin_width=1.0,
            condition_specific_encoding=True)

    elif model_name == 'efficient_encoding':
        from bauer.numerosity import EfficientEncodingEstimationModel
        model = EfficientEncodingEstimationModel(
            paradigm, grid_resolution=grid_resolution,
            n_min=10, n_max=40, response_bin_width=1.0)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Building model: {model_name} (grid_resolution={grid_resolution})")
    model.build_estimation_model(paradigm, hierarchical=False, flat_prior=False,
                                 save_p_choice=True)

    print(f"Free parameters: {list(model.free_parameters.keys())}")
    print(f"Sampling: {chains} chains, {tune} tune + {draws} draws, "
          f"target_accept={target_accept}")

    idata = model.sample(draws=draws, tune=tune, target_accept=target_accept,
                         chains=chains, cores=1)

    return model, idata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int, help='Subject ID')
    parser.add_argument('--model', default='log_encoding',
                        choices=['log_encoding', 'flexible_shared',
                                 'flexible_condition', 'efficient_encoding'],
                        help='Model to fit')
    parser.add_argument('--grid-resolution', type=int, default=31)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--target-accept', type=float, default=0.95)
    parser.add_argument('--output-dir', default='results/numerosity')
    args = parser.parse_args()

    paradigm = get_paradigm(args.subject)
    model, idata = fit_model(paradigm, args.model, args.grid_resolution,
                             args.draws, args.tune, args.chains,
                             args.target_accept)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    fn = os.path.join(args.output_dir,
                      f'sub-{args.subject:02d}_model-{args.model}_'
                      f'grid-{args.grid_resolution}.netcdf')
    idata.to_netcdf(fn)
    print(f"Saved to {fn}")

    # Print summary
    import arviz as az
    print("\nPosterior summary:")
    # Only print the main parameters (not encoding increments)
    var_names = [k for k in model.free_parameters.keys()
                 if not k.startswith('enc_')]
    print(az.summary(idata, var_names=var_names))


if __name__ == '__main__':
    main()
