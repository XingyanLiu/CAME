# -*- coding: UTF-8 -*-
"""
@author: Xingyan Liu
@file: _get_example_data.py
@time: 2021-06-12
"""

import os
from pathlib import Path
from typing import Sequence, Union, Mapping, List, Optional  # , Callable
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import logging

ROOT = Path(os.path.join(os.path.dirname(__file__), '..'))


def load_example_data():
    datadir = ROOT / 'sample_data'
    sp1, sp2 = ('human', 'mouse')
    dsnames = ('Baron_human', 'Baron_mouse')

    df_varmap_1v1 = pd.read_csv(datadir / f'gene_matches_1v1_{sp1}2{sp2}.csv', )
    df_varmap = pd.read_csv(datadir / f'gene_matches_{sp1}2{sp2}.csv', )

    dsn1, dsn2 = dsnames
    adata_raw1 = sc.read_h5ad(datadir / f'raw-{dsn1}.h5ad')
    adata_raw2 = sc.read_h5ad(datadir / f'raw-{dsn2}.h5ad')

    key_class = 'cell_ontology_class'
    # time_tag = make_nowtime_tag()
    example_dict = {
        'adatas': [adata_raw1, adata_raw2],
        'varmap': df_varmap,
        'varmap_1v1': df_varmap_1v1,
        'dataset_names': dsnames,
        'key_class': key_class,
    }
    logging.info(example_dict.keys())
    logging.debug(example_dict)
    return example_dict


if __name__ == '__main__':
    # logging.debug(
    #     logging.DEBUG,
    #     format=
    # )
    d = load_example_data()
    print(d)