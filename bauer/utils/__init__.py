def cluster_offers(d, n=6, key='log(risky/safe)'):
    return pd.qcut(d[key], n, duplicates='drop').apply(lambda x: x.mid)