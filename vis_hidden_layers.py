# -*- coding: UTF-8 -*-
"""
@author: Xingyan Liu
@file: vis_hidden_layers.py
@time: 2021-06-12
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
TEST_ON_MAC = not True
if not TEST_ON_MAC:
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
        ('testis_human', 'testis_monkey'),
    ]
    dsnames = DATASET_PAIRS[2]  # [::-1]
    dsn1, dsn2 = dsnames

    from DATASET_NAMES import Tissues, NAMES_ALL

    for _tiss in Tissues:
        NameDict = NAMES_ALL[_tiss]
        species = list(NameDict.keys())
        pair_species = CAME.base.make_pairs_from_lists(species, species)
        for _sp1, _sp2 in pair_species:
            if dsn1 in NameDict[_sp1] and dsn2 in NameDict[_sp2]:
                tiss, (sp1, sp2) = _tiss, (_sp1, _sp2)
                break

    print(f'Tissue:\t{tiss}', f'ref: {sp1}\t{dsn1}', f'que: {sp2}\t{dsn2}',
          sep='\n')
    datadir = Path('E:/lxy_pro/004/datasets') / 'formal' / tiss
    dir_gmap = Path(
        'E:/lxy_pro/004/resources/mart_exports/exported_gene_matches')
else:
    datadir = Path(os.path.abspath(__file__)).parent / 'CAME/sample_data'
    dir_gmap = datadir
    tiss = 'pancreas'
    sp1, sp2 = ('human', 'mouse')
    dsnames = ('Baron_human', 'Baron_mouse')

# In[]
dsn1, dsn2 = dsnames
df_varmap_1v1 = pd.read_csv(dir_gmap / f'gene_matches_1v1_{sp1}2{sp2}.csv', )
df_varmap = pd.read_csv(dir_gmap / f'gene_matches_{sp1}2{sp2}.csv', )

# setting directory for results
_time_tag = CAME.make_nowtime_tag()
subdir_res0 = f"{tiss}-{dsnames}{_time_tag}"

resdir = Path('./_case_res') / subdir_res0
figdir = resdir / 'figs'
sc.settings.figdir = figdir
CAME.check_dirs(figdir)

# In[]
# loading data
key_class1 = ['major_class', 'cell_ontology_class'][1]
key_class2 = key_class1

adata_raw1 = sc.read_h5ad(datadir / f'raw-{dsn1}.h5ad')
adata_raw2 = sc.read_h5ad(datadir / f'raw-{dsn2}.h5ad')
adatas = [adata_raw1, adata_raw2]

# In[]
''' subsampling and filtering genes
'''
for _adt, _name in zip([adata_raw1, adata_raw2], dsnames):
    if _adt.shape[0] >= 2e4:
        print(f'Doing subsampling for {_name}')
        sc.pp.subsample(_adt, fraction=0.5)

sc.pp.filter_genes(adata_raw1, min_cells=3)
sc.pp.filter_genes(adata_raw2, min_cells=3)

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
    n_pass=100,
    params_model=dict(residual=True)
)
load_other_ckpt = False
if load_other_ckpt:
    obs, df_probs, h_dict = pipeline.gather_came_results(
            dpair,
            trainer,
            classes=ENV_VARs['classes'],
            keys=(key_class1, key_class2),
            keys_compare=(key_class1, key_class2),
            resdir=resdir,
            checkpoint='last',
    )

# In[]
# ======================= further analysis =======================
obs_ids1, obs_ids2 = dpair.obs_ids1, dpair.obs_ids2
obs = dpair.obs
classes = dpair.classes
if 'unknown' in classes:
    classes = classes[: -1]
# h_dict = trainer.model.get_hidden_states()
adt = pp.make_adata(h_dict['cell'], obs=dpair.obs, assparse=False)
gadt = pp.make_adata(h_dict['gene'], obs=dpair.var, assparse=False)



