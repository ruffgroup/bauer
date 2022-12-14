import pandas as pd
import pkg_resources
import numpy as np

def load_garcia2022():
    """Return a dataframe containing the behavioral data from the Magnitude task
    in Baretto Garcia et al., 2022


    """
    stream = pkg_resources.resource_stream(__name__, '../data/garcia2022.csv')
    df = pd.read_csv(stream, index_col=[0, 1, 2])
    df['log(n2/n1)'] = np.log(df['n2'] / df['n1'])
    # df['trial_nr'] = df.groupby(['subject'], group_keys=False).apply(lambda d: pd.Series(np.arange(len(d))+1, index=d.index))
    return df
