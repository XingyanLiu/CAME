# -*- coding: UTF-8 -*-
"""
@author: Xingyan Liu
@file: case-supervised.py
@time: 2021-05-30
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

# ROOT = os.path.dirname(os.path.abspath(__file__))
# os.chdir(ROOT)

import CAME
from CAME.pipeline_supervised import preprocess_unaligned, main_for_unaligned
from CAME import pipeline_supervised, pp, pl

# In[]
datadir = Path(os.path.abspath(__file__)).parent / 'CAME/sample_data'
sp1, sp2 = ('human', 'mouse')
dsnames = ('Baron_human', 'Baron_mouse')

df_varmap_1v1 = pd.read_csv(datadir / f'gene_matches_1v1_{sp1}2{sp2}.csv', )
df_varmap = pd.read_csv(datadir / f'gene_matches_{sp1}2{sp2}.csv', )

dsn1, dsn2 = dsnames
adata_raw1 = sc.read_h5ad(datadir / f'raw-{dsn1}.h5ad')
adata_raw2 = sc.read_h5ad(datadir / f'raw-{dsn2}.h5ad')
adatas = [adata_raw1, adata_raw2]

key_class = 'cell_ontology_class'
time_tag = CAME.make_nowtime_tag()
resdir = Path('_temp') / f'{dsnames}-{time_tag}'

# In[]
''' default pipeline of CAME
'''
came_inputs, (adata1, adata2) = preprocess_unaligned(
    adatas,
    key_class=key_class,
    use_scnets=False
)
n_epochs = 400
dpair, trainer, h_dict = main_for_unaligned(
        **came_inputs,
        df_varmap=df_varmap,
        df_varmap_1v1=df_varmap_1v1,
        dataset_names=dsnames,
        key_class1=key_class,
        key_class2=key_class,
        do_normalize=True,
        n_epochs=n_epochs,
        resdir=resdir,
        check_umap=not True,  # True for visualizing embeddings each 40 epochs
        n_pass=100,
    )

# In[]
''' ======================= further analysis =======================
'''
obs_ids1, obs_ids2 = dpair.obs_ids1, dpair.obs_ids2
obs = dpair.obs
classes = dpair.classes
if 'unknown' in classes:
    classes = classes[: -1]
# h_dict = trainer.model.get_hidden_states()
adt = pp.make_adata(h_dict['cell'], obs=dpair.obs, assparse=False)
gadt = pp.make_adata(h_dict['gene'], obs=dpair.var, assparse=False)

# adt.write(resdir / 'adt_hidden_cell.h5ad')
# gadt.write(resdir / 'adt_hidden_gene.h5ad')

# In[]
'''======================= cell embeddings ======================='''
# from CAME_v0.utils.plot_pub import plot_pure_umap

sc.set_figure_params(dpi_save=200)

sc.pp.neighbors(adt, n_neighbors=15, metric='cosine', use_rep='X')
sc.tl.umap(adt)
sc.pl.umap(adt, color=['dataset', 'celltype'], ncols=1)
# setting UMAP to the original adata
obs_umap = adt.obsm['X_umap']
adata1.obsm['X_umap'] = obs_umap[obs_ids1]
adata2.obsm['X_umap'] = obs_umap[obs_ids2]

ftype = ['.svg', ''][1]
sc.pl.umap(adt, color='dataset', save=f'-dataset{ftype}')
sc.pl.umap(adt, color='celltype', save=f'-ctype{ftype}')

adt.write(resdir / 'adt_hidden_cell.h5ad')

# In[]
'''===================== gene embeddings ====================='''
sc.set_figure_params(dpi_save=200)

sc.pp.neighbors(gadt, n_neighbors=15, metric='cosine', use_rep='X')
sc.tl.umap(gadt)
sc.pl.umap(gadt, color='dataset', )

''' joint gene module extraction '''
sc.tl.leiden(gadt, resolution=.8, key_added='module')
sc.pl.umap(gadt, color=['dataset', 'module'], ncols=1, save=f'combined.pdf')

''' link-weights between homologous gene pairs '''
df_var_links = CAME.weight_linked_vars(
    gadt.X, dpair._vv_adj, names=dpair.get_vnode_names(),
    matric='cosine', index_names=dsnames,
)

# split
gadt1, gadt2 = pp.bisplit_adata(gadt, 'dataset', dsn1, reset_index_by='name')
color_by = 'module'
sc.pl.umap(gadt1, color=color_by, s=10, edges=True, edges_width=0.05,
           save=f'_{color_by}-{dsn1}')
sc.pl.umap(gadt2, color=color_by, s=10, edges=True, edges_width=0.05,
           save=f'_{color_by}-{dsn2}')