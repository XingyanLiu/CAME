# -*- coding: UTF-8 -*-
"""
@CreateDate: 2021/06/23
@Author: Xingyan Liu
@File: test_uk_identify.py
@Project: CAME
"""


import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import scanpy as sc
from scipy import sparse
from scipy.special import softmax

import networkx as nx
import torch

import CAME
from CAME import pipeline, pp, pl

# In[]
DATASET_PAIRS = [
    ('zebrafish_LD_10hr', 'mouse_NMDA_24hr'),
    ('panc8', 'Baron_mouse'),
    ('Baron_human', 'Baron_mouse'),
    ('Baron_human', 'FACS'),
    ('Lake_2018', 'Tasic18'),
    ('Lake_2018', 'Tosches_turtle'),
    ('Lake_2018', 'Tosches_lizard'),
    ('Tosches_turtle', 'Tosches_lizard'),
    ('testis_human', 'testis_mouse'),
    ('testis_human', 'testis_mouse0'),
    ('testis_human', 'testis_monkey'),
]
dsnames = DATASET_PAIRS[4]  # [5]
dsn1, dsn2 = dsnames

from DATASET_NAMES import Tissues, NAMES_ALL

for _tiss in Tissues:
    NameDict = NAMES_ALL[_tiss]
    species = list(NameDict.keys())
    pair_species = CAME.base.make_pairs_from_lists(species, species)
    for _sp1, _sp2 in pair_species:
        if dsn1 in NameDict[_sp1] and dsn2 in NameDict[_sp2] + ['testis_mouse0']:
            tiss, (sp1, sp2) = _tiss, (_sp1, _sp2)
            break

print(f'Tissue:\t{tiss}', f'ref: {sp1}\t{dsn1}', f'que: {sp2}\t{dsn2}',
      sep='\n')

# In[]
datadir = Path('E:/lxy_pro/004/datasets')
dir_gmap = Path('E:/lxy_pro/004/resources/mart_exports/exported_gene_matches')

# tiss, (sp1, sp2) = 'brain', ('human', 'lizard')
# dsnames = ('Lake_2018', 'Tosches_lizard')
# dsn1, dsn2 = dsnames
dir_formal = datadir / 'formal' / tiss
df_varmap_1v1 = pd.read_csv(dir_gmap / f'gene_matches_1v1_{sp1}2{sp2}.csv', )
df_varmap = pd.read_csv(dir_gmap / f'gene_matches_{sp1}2{sp2}.csv', )

_time_tag = CAME.make_nowtime_tag()
resdir0 = Path('./_case_res') / f"uk-{dsnames}{_time_tag}"

# In[]
''' loading data
'''
key_class1 = ['major_class', 'cell_ontology_class'][1]
key_class2 = key_class1

adata_raw10 = sc.read_h5ad(dir_formal / f'raw-{dsn1}.h5ad')
adata_raw2 = sc.read_h5ad(dir_formal / f'raw-{dsn2}.h5ad')
adatas = [adata_raw10, adata_raw2]

# subsample cells
for _adt, _name in zip([adata_raw10, adata_raw2], dsnames):
    if _adt.shape[0] >= 2e4:
        print(f'Doing subsampling for {_name}')
        sc.pp.subsample(_adt, fraction=0.5)

sc.pp.filter_genes(adata_raw10, min_cells=3)
sc.pp.filter_genes(adata_raw2, min_cells=3)


all_types = list(
    set(adata_raw10.obs[key_class1]).intersection(adata_raw2.obs[key_class1]))
# In[]
type_rm = all_types[0]
for type_rm in all_types:
    resdir = resdir0 / f'rm-{type_rm}'
    figdir = resdir / 'figs'
    sc.settings.figdir = figdir
    CAME.check_dirs(figdir)

    # remove known-ref-type
    adata_raw1 = pp.remove_adata_groups(
        adata_raw10, key_class1, type_rm, copy=True)
    adatas = [adata_raw1, adata_raw2]

    # In[]
    ''' default pipeline of CAME
    '''
    came_inputs, (adata1, adata2) = pipeline.preprocess_unaligned(
        adatas,
        key_class=key_class1,
        use_scnets=True,
    )

    dpair, trainer, h_dict, ENV_VARs = pipeline.main_for_unaligned(
        **came_inputs,
        df_varmap=df_varmap,
        df_varmap_1v1=df_varmap_1v1,
        dataset_names=dsnames,
        key_class1=key_class1,
        key_class2=key_class2,
        do_normalize=True,
        n_epochs=350,
        resdir=resdir,
        check_umap=not True,  # True for visualizing embeddings each 40 epochs
        n_pass=200,
        params_model=dict(residual=False)
    )
    del trainer.model, trainer, h_dict
    
    # In[]
    

    # obs_ids1, obs_ids2 = dpair.obs_ids1, dpair.obs_ids2

    # df_logits2 = pd.read_csv(resdir / 'df_logits2.csv', index_col=0)
    # predictor = CAME.Predictor.load(resdir / 'predictor.json')
    # pred_test = predictor.predict(
    #     df_logits2.values, p=1e-4, trans_mode=3)

    # pl.plot_co
    
    
    
    