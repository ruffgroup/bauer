import pandas as pd
import pkg_resources
import numpy as np

def load_garcia2022(task='magnitude', remove_non_responses=True):
    """Return a dataframe containing the behavioral data from the Magnitude task
    in Baretto Garcia et al., 2022


    """

    if task=='magnitude':
        fn = '../data/garcia2022_magnitude.csv'
    elif task=='risk':
        fn = '../data/garcia2022_risk.csv'
    
    stream = pkg_resources.resource_stream(__name__, fn)
    df = pd.read_csv(stream, index_col=[0, 1, 2, 3])

    if remove_non_responses:
        df = df[~df['choice'].isnull()]
        df['choice'] = df['choice'].astype(bool)
    # df['log(n2/n1)'] = np.log(df['n2'] / df['n1'])
    # df['trial_nr'] = df.groupby(['subject'], group_keys=False).apply(lambda d: pd.Series(np.arange(len(d))+1, index=d.index))
    return df
