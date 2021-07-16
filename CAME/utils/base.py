# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 12:16:32 2021

@author: Xingyan Liu
=======================================
    Basic functions

"""

import os
import logging
from typing import Sequence, Union, Mapping
import json
import pickle
import time
import datetime
import numpy as np
import pandas as pd
from scipy import sparse


def strline(seq):
    return '\n'.join(map(str, seq))


def _upper_strs(strs):
    return [s.upper() for s in strs]


def _lower_strs(strs):
    return [s.lower() for s in strs]


def _capital_strs(strs):
    return [s.capitalize() for s in strs]


# In[]
""" I/O functions
    handling AnnData object(s)
"""


def save_pickle(obj, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)
    print('object saved into:\n\t', fpath)


def load_pickle(fp):
    """
    >>> load_pickle('results.pickle')
    """
    with open(fp, 'rb') as f:
        res = pickle.load(f)
    return res


def check_dirs(path):
    if os.path.exists(path):
        print('already exists:\n\t%s' % path)
    else:
        os.makedirs(path)
        print('a new directory made:\n\t%s' % path)


def write_info(fn, **dicts):
    """
    key words parameter-dicts
    """
    f = open(fn, 'w')
    logging.info('file name:\n\t', fn, file=f)
    for kw, val in dicts.items():
        if isinstance(val, dict):
            dict_str = strline(val.items())
            print(f'\n>{kw}:\n{dict_str}', file=f)
        else:
            print(f'\n>{kw}:\n', val, file=f)
    f.close()


def save_json_dict(dct, fname='test_json.json', encoding='utf-8'):
    with open(fname, 'w', encoding=encoding) as jsfile:
        json.dump(dct, jsfile, ensure_ascii=False)
    logging.info(fname)


def load_json_dict(fname, encoding='utf-8'):
    with open(fname, encoding=encoding) as f:
        dct = json.load(f)
    return dct


def make_nowtime_tag(nowtime=None, brackets=True):
    if nowtime is None:
        nowtime = datetime.datetime.today()
    d = nowtime.strftime('%m-%d')
    t = str(nowtime.time()).split('.')[0].replace(':', '.')
    if brackets:
        fmt = '({} {})'
    else:
        fmt = '{} {}'
    return fmt.format(d, t)


def dec_timewrapper(tag='function'):
    def my_decorator(foo):
        def wrapper(*args, **kargs):
            t0 = time.time()
            res = foo(*args, **kargs)
            t1 = time.time()
            print(f'[{tag}] Time used: {t1 - t0: .4f} s')
            return res

        return wrapper

    return my_decorator


def make_pairs_from_lists(lst1, lst2=None, inverse=False, skip_equal=True):
    """ making combinational pairs
    """
    lst2 = lst1 if lst2 is None else lst2
    pairs = []
    for x in lst1:
        for y in lst2:
            if skip_equal and x == y: continue
            pairs.append((x, y))
            if inverse:
                pairs.append((y, x))
    return pairs


# In[]
def subsample_single(N,
                     frac=0.25, n_min=50,
                     n_out=None,
                     seed=0):
    """
    N:
        The total number of the original indices to be subsampled
    frac: 
        Subsample to this `fraction` of the number of indices.
    n_min:
        If the resulting number of indices is less than `n_min`, then 
        `min([n_min, N])` indices will be sampled, where `N` is 
        the total number of the original indices.
    n_out:
        if specified, it must be smaller than N, and the two arguments
        `frac` and `n_min` will be ignored.
    seed:
        random state
    """
    #    N = len(ids)
    if n_out is None:
        n_ids_sub = max([np.ceil(N * frac), n_min])
        except_error = False
    else:
        n_ids_sub = n_out
        except_error = True
    if n_ids_sub >= N:
        if except_error:
            raise ValueError('The argument `n_out` should be smaller than the '
                             f'length of the input `ids`, got {n_out} >= {N}.')
        print('no subsampling was done.')
        return np.arange(N)
    else:
        np.random.seed(seed)
        _ids_sub = np.random.choice(N, size=n_ids_sub, replace=False)

        return _ids_sub


def subsample_each_group(
        group_labels,
        n_out=50,
        seed=0,
):
    """
    randomly sample indices from each group, labeled by `group_labels`.
    and return the sampled indices.
    
    n_out: number of samples for each group
    
    """
    np.random.seed(seed)
    if isinstance(group_labels, pd.Series):
        ids_all = group_labels.index
    else:
        ids_all = np.arange(len(group_labels))
    res_ids = []

    for lb in pd.unique(group_labels):
        ids = np.flatnonzero(group_labels == lb)
        if len(ids) <= n_out:
            ids_sub = ids
        else:
            ids_sub = np.take(
                ids,
                np.random.choice(len(ids), size=n_out, replace=False)
            )
        res_ids.append(ids_sub)
    #    print(list(map(len, res_ids)))
    res_ids = np.hstack(res_ids)

    return np.take(ids_all, sorted(res_ids))
