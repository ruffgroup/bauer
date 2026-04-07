#!/usr/bin/env python
"""Fit efficient coding estimation models via MCMC.

Usage:
    # Single subject:
    python fit_efficient_coding.py 5 --model perception

    # Hierarchical (all subjects):
    python fit_efficient_coding.py --hierarchical --model perception

Saves arviz InferenceData as netcdf.
"""

import argparse
import os
import numpy as np
import pandas as pd


def get_paradigm(subject_ids=None, bids_folder='/data/ds-abstract_values_pilot'):
    """Load and prepare paradigm (both mappings per subject)."""
    from bauer.utils.data import load_abstract_values_pilot

    df = load_abstract_values_pilot(bids_folder=bids_folder, subjects=subject_ids)

    all_paradigms = []
    for sub_id in df.index.get_level_values('subject').unique():
        sub_data = df.xs(sub_id, level='subject').reset_index()
        p = sub_data[['orientation', 'response', 'mapping']].copy()
        p['subject'] = sub_id
        all_paradigms.append(p)

    paradigm = pd.concat(all_paradigms, ignore_index=True)
    paradigm = paradigm.set_index(['subject', paradigm.groupby('subject').cumcount()])
    paradigm.index.names = ['subject', 'trial']

    subjects = paradigm.index.get_level_values('subject').unique()
    print(f"Loaded {len(subjects)} subjects, {len(paradigm)} trials")
    for m in paradigm['mapping'].unique():
        print(f"  {m}: {(paradigm['mapping'] == m).sum()} trials")

    return paradigm


def make_model(paradigm, model_name, grid_resolution):
    """Create model instance."""
    if model_name == 'perception':
        from bauer.efficient_coding import EfficientPerceptionModel
        return EfficientPerceptionModel(
            paradigm, grid_resolution=grid_resolution,
            perceptual_prior='long_term')

    elif model_name == 'valuation':
        from bauer.efficient_coding import EfficientValuationModel
        return EfficientValuationModel(
            paradigm, grid_resolution=grid_resolution)

    elif model_name == 'sequential':
        from bauer.efficient_coding import SequentialEfficientCodingModel
        return SequentialEfficientCodingModel(
            paradigm, grid_resolution=grid_resolution,
            perceptual_prior='long_term')

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
    parser.add_argument('--model', default='perception',
                        choices=['perception', 'valuation', 'sequential'])
    parser.add_argument('--grid-resolution', type=int, default=31)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--target-accept', type=float, default=0.95)
    parser.add_argument('--nuts-sampler', default='pymc',
                        choices=['pymc', 'numpyro', 'nutpie'])
    parser.add_argument('--bids-folder', default='/data/ds-abstract_values_pilot')
    parser.add_argument('--output-dir', default='results/efficient_coding')
    args = parser.parse_args()

    if args.hierarchical:
        subject_ids = args.subjects  # None = all
        paradigm = get_paradigm(subject_ids, args.bids_folder)
        hierarchical = True
        label = 'hierarchical'
        if subject_ids:
            label += f'_n{len(subject_ids)}'
    else:
        if args.subject is None:
            parser.error("Provide a subject ID, or use --hierarchical")
        paradigm = get_paradigm([args.subject], args.bids_folder)
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

    os.makedirs(args.output_dir, exist_ok=True)
    fn = os.path.join(args.output_dir,
                      f'{label}_model-{args.model}_grid-{args.grid_resolution}.netcdf')
    idata.to_netcdf(fn)
    print(f"Saved to {fn}")

    import arviz as az
    var_names = list(model.free_parameters.keys())
    if hierarchical:
        var_names = [f'{k}_mu' for k in var_names] + [f'{k}_sd' for k in var_names]
    print("\nPosterior summary:")
    print(az.summary(idata, var_names=var_names))


if __name__ == '__main__':
    main()
