#!/usr/bin/env python
"""Fit efficient coding estimation models via MCMC.

Usage:
    python fit_efficient_coding.py SUBJECT [--model MODEL] [--grid-resolution N]
                                   [--draws N] [--tune N] [--chains N]
                                   [--bids-folder DIR] [--output-dir DIR]

Fits the specified model to a single subject's data (both CDF and inverse-CDF
mapping conditions jointly) using MCMC sampling.
"""

import argparse
import os
import numpy as np
import pandas as pd


def get_paradigm(subject_id, bids_folder):
    """Load and prepare paradigm for a single subject (both mappings)."""
    from bauer.utils.data import load_abstract_values_pilot

    df = load_abstract_values_pilot(bids_folder=bids_folder,
                                     subjects=[subject_id])

    sub_data = df.xs(subject_id, level='subject').reset_index()
    paradigm = sub_data[['orientation', 'response', 'mapping']].copy()
    paradigm.index = pd.MultiIndex.from_arrays(
        [np.full(len(paradigm), subject_id), range(len(paradigm))],
        names=['subject', 'trial'])

    print(f"Subject {subject_id}: {len(paradigm)} trials "
          f"({(paradigm['mapping']=='cdf').sum()} CDF, "
          f"{(paradigm['mapping']=='inverse_cdf').sum()} inv-CDF)")

    return paradigm


def fit_model(paradigm, model_name, grid_resolution, draws, tune, chains,
              target_accept):
    """Build and sample the specified model."""

    if model_name == 'perception':
        from bauer.efficient_coding import EfficientPerceptionModel
        model = EfficientPerceptionModel(
            paradigm, grid_resolution=grid_resolution,
            perceptual_prior='long_term')

    elif model_name == 'valuation':
        from bauer.efficient_coding import EfficientValuationModel
        model = EfficientValuationModel(
            paradigm, grid_resolution=grid_resolution)

    elif model_name == 'sequential':
        from bauer.efficient_coding import SequentialEfficientCodingModel
        model = SequentialEfficientCodingModel(
            paradigm, grid_resolution=grid_resolution,
            perceptual_prior='long_term')

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
    parser.add_argument('--model', default='perception',
                        choices=['perception', 'valuation', 'sequential'],
                        help='Model to fit')
    parser.add_argument('--grid-resolution', type=int, default=31)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--target-accept', type=float, default=0.95)
    parser.add_argument('--bids-folder', default='/data/ds-abstract_values_pilot')
    parser.add_argument('--output-dir', default='results/efficient_coding')
    args = parser.parse_args()

    paradigm = get_paradigm(args.subject, args.bids_folder)
    model, idata = fit_model(paradigm, args.model, args.grid_resolution,
                             args.draws, args.tune, args.chains,
                             args.target_accept)

    os.makedirs(args.output_dir, exist_ok=True)
    fn = os.path.join(args.output_dir,
                      f'sub-{args.subject:02d}_model-{args.model}_'
                      f'grid-{args.grid_resolution}.netcdf')
    idata.to_netcdf(fn)
    print(f"Saved to {fn}")

    import arviz as az
    print("\nPosterior summary:")
    print(az.summary(idata, var_names=list(model.free_parameters.keys())))


if __name__ == '__main__':
    main()
