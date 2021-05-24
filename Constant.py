#!/usr/bin/env python

import os
import numpy as np
import scipy.io as sio
import gzip
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

FULL, SAMPLE, KEEPK = 'FULL', 'SAMPLE', 'KEEPK'
GEX, WES, CNV, METH = 'GEX', 'WES', 'CNV', 'MET'
PPO, KRL, LKRL, KBMTL, KRR, RF, EN = 'PPORank', 'KRL', 'LKRL', 'KBMTL', 'KRR', 'RF', 'EN'
BASELINES = [KRL, EN, KRR]
METHODS = BASELINES + [KBMTL, LKRL, KRL]
PARAM_STR, RANK_STR, PERCENTILE_STR, NDCG_STR, PRECISION_STR = 'PARAM', 'RANK', 'PERCENTILE', 'NDCG', 'PRECISION'
DELIM = ' '

# DRUGS
DOCETAXEL = 'docetaxel'
BORTEZOMIB = 'bortezomib'
LAPATINIB = 'lapatinib'  # 拉帕替尼
VELIPARIB = 'veliparib'
OLAPARIB1 = 'olaparib1'
OLAPARIB2 = 'olaparib2'
TALAZOPARIB = 'talazoparib'
RUCAPARIB = 'rucaparib'
RUXOLITINIB = 'ruxolitinib'

DRUG_NAMES = {DOCETAXEL: DOCETAXEL,
              BORTEZOMIB: BORTEZOMIB,
              LAPATINIB: LAPATINIB,
              VELIPARIB: 'abt-888',
              OLAPARIB1: 'olaparib',
              OLAPARIB2: 'olaparib (rescreen)',
              TALAZOPARIB: 'bmn-673',
              RUCAPARIB: 'ag-014699',
              RUXOLITINIB: RUXOLITINIB}

DRUG_IDS = {DOCETAXEL: 1007,
            BORTEZOMIB: 104,
            LAPATINIB: 119,
            VELIPARIB: 1018,
            OLAPARIB1: 1017,
            OLAPARIB2: 1495,
            TALAZOPARIB: 1259,
            RUCAPARIB: 1175,
            RUXOLITINIB: 206}

PARP_INHIBS = [VELIPARIB, OLAPARIB1, OLAPARIB2, TALAZOPARIB, RUCAPARIB]
TNBC_DRUGS = [RUXOLITINIB] + PARP_INHIBS


def raise_exception(*messages):
    # print (>> sys.stderr, 'ERROR:', ' '.join(map(str, messages))
    raise Exception


def open_file(filename, mode='r', compresslevel=9):
    if mode not in ['r', 'rb', 'a', 'ab', 'w', 'wb']:
        raise_exception('file mode not supported:', mode)
    if filename.endswith('.gz') or (not os.path.exists(filename) and os.path.exists(filename + '.gz')):
        # gzip automatically adds 'b' to the 'r', 'a', and 'w' modes
        return gzip.open(filename if filename.endswith('.gz') else filename + '.gz', mode, compresslevel)
    else:
        return open(filename, mode)


def line_split_gen(filename, delim='\t', strip='\n', comment='#', skip_rows=0, skip_columns=0):
    with open_file(filename) as f:
        for _ in range(skip_rows):
            f.readline()
        for line in f:
            if comment is not None and line.startswith(comment):
                continue
            line_split = line.strip(strip).split(delim)
            yield line_split[skip_columns:]


def cm_to_inch(value):
    return float(value) / 2.54


def intersect_index(list1, list2):
    '''
    Given two lists find the index of intersect in list1

    Parameterst
    ----------
    list1: 1d numpy array
    list2: 1d numpy array
    '''
    intersect = np.intersect1d(list1, list2)
    intersect = pd.DataFrame(intersect, columns=['id'])
    list1 = np.vstack([np.arange(list1.shape[0]), list1]).T
    list1 = pd.DataFrame(list1, columns=['index1', 'id'])
    list2 = np.vstack([np.arange(list2.shape[0]), list2]).T
    list2 = pd.DataFrame(list2, columns=['index2', 'id'])
    merged = pd.merge(list1, intersect, on='id', how='right')
    merged = pd.merge(merged, list2, on='id', how='left')

    return merged


def load_pRRophetic_y(filename):
    y_df = pd.read_csv(filename)
    return y_df.loc[:, 'Resp'].values


def load_pRRophetic_data(
        data_dir, fn_prefix, train_suffix='_trainFrame.csv.gz', test_suffix='_testFrame.csv.gz',
        test_y_suffix='_test_y.csv.gz', verbose=True):
    train_df = pd.read_csv(os.path.join(data_dir, fn_prefix + train_suffix))
    test_df = pd.read_csv(os.path.join(data_dir, fn_prefix + test_suffix))
    assert np.all(train_df.columns[2:] == test_df.columns[1:])
    assert train_df.columns[1] in ['Resp', 'trainPtyle']

    train_X = train_df.iloc[:, 2:].values
    train_samples = np.array(train_df.iloc[:, 0])
    train_y = train_df.iloc[:, 1].values

    test_X = test_df.iloc[:, 1:].values
    test_samples = np.array(test_df.iloc[:, 0])
    assert list(np.array(pd.read_csv(os.path.join(
        data_dir, fn_prefix + '_preds.csv.gz')).iloc[:, 0])) == list(test_samples)
    test_y = load_pRRophetic_y(os.path.join(data_dir, fn_prefix + test_y_suffix)) if test_y_suffix is not None else None

    if verbose:
        print('pRRophetic train: X {} y {} samples {}'.format(train_X.shape, train_y.shape, train_samples.shape))
    if verbose:
        print('pRRophetic test: X {} y {} samples {}'.format(test_X.shape,
                                                             test_y.shape if test_y is not None else None, test_samples.shape))

    assert train_X.shape[0] == train_samples.shape[0] and train_X.shape[0] == train_y.shape[0]
    assert test_X.shape[0] == test_samples.shape[0] and (test_y is None or test_X.shape[0] == test_y.shape[0])

    return train_X, train_y, train_samples, test_X, test_y, test_samples


def get_median_trick_gamma(X, Y=None):
    if Y is None:
        Y = X
    distances = euclidean_distances(X, Y)
    squared_distances = distances.flatten() ** 2
    gamma = 1.0 / np.median(squared_distances)
    return gamma
