import pandas as pd
import pkg_resources

def load_garcia2022():
    """Return a dataframe about the 68 different Roman Emperors.


    """
    stream = pkg_resources.resource_stream(__name__, '../data/garcia2022.csv')
    df = pd.read_csv(stream, index_col=[0, 1, 2])
    # df['trial_nr'] = df.groupby(['subject'], group_keys=False).apply(lambda d: pd.Series(np.arange(len(d))+1, index=d.index))
    return df
