from __future__ import absolute_import, division, print_function

import pandas

from util.data_set_helpers import DataSets, DataSet

def read_data_sets(train_csvs, dev_csvs, test_csvs,
                   train_batch_size, dev_batch_size, test_batch_size,
                   numcep, numcontext, thread_count=8,
                   stride=1, offset=0, next_index=lambda s, i: i + 1,
                   limit_dev=0, limit_test=0, limit_train=0, sets=[]):
    # Read the processed set files from disk
    def read_csvs(csvs):
        files = None
        for csv in csvs:
            file = pandas.read_csv(csv)
            if files is None:
                files = file
            else:
                files = files.append(file)
        return files

    train_files = read_csvs(train_csvs)
    dev_files = read_csvs(dev_csvs)
    test_files = read_csvs(test_csvs)

    # Create train DataSet from all the train archives
    train = None
    if train_files is not None and "train" in sets:
        train = _read_data_set(train_files, thread_count, train_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('train', i), limit=limit_train)

    # Create dev DataSet from all the dev archives
    dev = None
    if dev_files is not None and "dev" in sets:
        dev = _read_data_set(dev_files, thread_count, dev_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('dev', i), limit=limit_dev)

    # Create test DataSet from all the test archives
    test = None
    if test_files is not None and "test" in sets:
        test = _read_data_set(test_files, thread_count, test_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('test', i), limit=limit_test)

    # Return DataSets
    return DataSets(train, dev, test)

def _read_data_set(filelist, thread_count, batch_size, numcep, numcontext, stride=1, offset=0, next_index=lambda i: i + 1, limit=0):
    # Optionally apply dataset size limits
    if limit > 0:
        filelist = filelist.iloc[:limit]

    filelist = filelist[offset::stride]

    # Return DataSet
    return DataSet(filelist, thread_count, batch_size, numcep, numcontext, next_index=next_index)
