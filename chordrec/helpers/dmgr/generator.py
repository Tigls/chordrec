import random
def iterate_aggregated_data(aggregated_data_source, batch_size, randomise=False, expand=False):

    n_ds = aggregated_data_source.n_datasources
    ds_idxs = list(range(n_ds))

    if randomise:
        random.shuffle(ds_idxs)
    index = -1
    while index < len(ds_idxs):
        index += 1
        if index == len(ds_idxs) - 1:
            index = 0
        ds = aggregated_data_source.datasource(ds_idxs[index])
        idxs = range(ds.n_data)
        start_idx = 0
        while start_idx < len(ds):
            batch_idxs = list(idxs[start_idx:start_idx + batch_size])

            # last batch could be too small
            if len(batch_idxs) < batch_size and expand:
                # fill up with random indices not yet in the set
                n_missing = batch_size - len(batch_idxs)
                batch_idxs += list(random.sample(idxs[:start_idx], n_missing))

            start_idx += batch_size
            yield ds[batch_idxs]

