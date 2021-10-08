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

CAME_ROOT = Path(os.path.join(os.path.dirname(__file__), '..'))


def _extract_zip(
        fp_zip=CAME_ROOT / 'sample_data.zip',
        fp_unzip=CAME_ROOT / 'sample_data',
):
    import zipfile
    with zipfile.ZipFile(fp_zip) as zipf:
        zipf.extractall(fp_unzip)


def load_example_data():
    datadir = CAME_ROOT / 'sample_data'

    sp1, sp2 = ('human', 'mouse')
    dsnames = ('Baron_human', 'Baron_mouse')
    dsn1, dsn2 = dsnames
    fp1, fp2 = datadir / f'raw-{dsn1}.h5ad', datadir / f'raw-{dsn2}.h5ad'
    if not fp1.exists():
        fp1 = datadir / f'raw-{dsn1}-sampled.h5ad'
    fp_varmap_1v1 = datadir / f'gene_matches_1v1_{sp1}2{sp2}.csv'
    fp_varmap = datadir / f'gene_matches_{sp1}2{sp2}.csv'

    if not (datadir.exists() and fp1.exists() and fp2.exists() and
            fp_varmap.exists() and fp_varmap_1v1.exists()):
        _extract_zip()

    df_varmap_1v1 = pd.read_csv(fp_varmap_1v1, )
    df_varmap = pd.read_csv(fp_varmap, )

    adata_raw1, adata_raw2 = sc.read_h5ad(fp1), sc.read_h5ad(fp2)

    key_class = 'cell_ontology_class'
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
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')
    d = load_example_data()
    print(d)
