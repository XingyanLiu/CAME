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
from CAME import pp, pl


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
    ('testis_human', 'testis_monkey'),
]
dsnames = DATASET_PAIRS[0]  # [::-1]
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

# In[]
datadir0 = Path('E:/lxy_pro/004/datasets')
dir_gmap = Path('E:/lxy_pro/004/resources/mart_exports/exported_gene_matches')

datadir = datadir0 / 'formal' / tiss
df_varmap_1v1 = pd.read_csv(dir_gmap / f'gene_matches_1v1_{sp1}2{sp2}.csv', )
df_varmap = pd.read_csv(dir_gmap / f'gene_matches_{sp1}2{sp2}.csv', )

_time_tag = CAME.make_nowtime_tag()
subdir_res0 = f"{tiss}-{dsnames}{_time_tag}"

resdir = Path('./_case_res') / subdir_res0
figdir = resdir / 'figs'
sc.settings.figdir = figdir
CAME.check_dirs(figdir)

key_class = 'cell_ontology_class'

# In[]
#datadir = Path(os.path.abspath(__file__)).parent / 'CAME/sample_data'
#sp1, sp2 = ('human', 'mouse')
#dsnames = ('Baron_human', 'Baron_mouse')

#df_varmap_1v1 = pd.read_csv(datadir / f'gene_matches_1v1_{sp1}2{sp2}.csv', )
#df_varmap = pd.read_csv(datadir / f'gene_matches_{sp1}2{sp2}.csv', )

#dsn1, dsn2 = dsnames
#
#key_class = 'cell_ontology_class'
#time_tag = CAME.make_nowtime_tag()
#resdir = Path('_temp') / f'{dsnames}-{time_tag}'

# In[]
adata_raw1 = sc.read_h5ad(datadir / f'raw-{dsn1}.h5ad')
adata_raw2 = sc.read_h5ad(datadir / f'raw-{dsn2}.h5ad')
adatas = [adata_raw1, adata_raw2]


# In[]
''' default pipeline of CAME
'''
came_inputs, (adata1, adata2) = preprocess_unaligned(
    adatas,
    key_class=key_class,
    use_scnets=False
)
n_epochs = 500
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
        model_params=dict(residual=False)
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
dsnames
cross_acc2 = (obs['predicted1'][obs_ids2] == obs['celltype'][obs_ids2]).sum() / len(obs_ids2)
cross_acc1 = (obs['predicted2'][obs_ids1] == obs['celltype'][obs_ids1]).sum() / len(obs_ids1)
print(f'{dsn1} to {dsn2} acc: {cross_acc2:.4f}')
print(f'{dsn2} to {dsn1} acc: {cross_acc1:.4f}')

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
sc.pl.umap(adt, color=['dataset', 'celltype'], save=f'-cell{ftype}')
sc.pl.umap(adt, color='dataset', save=f'-dataset{ftype}')
sc.pl.umap(adt, color='celltype', save=f'-ctype{ftype}')

adt.write(resdir / 'adt_hidden_cell.h5ad')

# In[]
''' similaraties of cell-type embeddings
'''
adt1, adt2 = pp.bisplit_adata(adt, 'dataset', dsn1, reset_index_by='original_name')
avg_embed1 = pp.group_mean_adata(adt1, 'celltype')
avg_embed2 = pp.group_mean_adata(adt2, 'celltype')

from scipy.spatial.distance import cdist
dist = cdist(avg_embed1.values.T, avg_embed2.values.T, metric='cosine')
sim = pd.DataFrame(
        data=1 - dist,
        index=avg_embed1.columns, columns=avg_embed2.columns
        )           
ax = pl.heatmap(sim, order_col=True, order_row=True, figsize=(5, 4), 
                fp=resdir / 'celltype_embed_sim.png')
ax.figure
#%matplotlib inline

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


# In[]
''' ============ cell type gene-profiles on gene embeddings ============
'''
# averaged expressions
avg_expr1 = pp.group_mean_adata(adata_raw1, groupby=key_class,
                                features=dpair.vnode_names1, use_raw=True)
avg_expr2 = pp.group_mean_adata(adata_raw2, groupby=key_class,
                                features=dpair.vnode_names2, use_raw=True)
# adata_raw1.X.data

avg_expr_add1, avg_expr_add2 = list(map(
    lambda x: pp.zscore(x.T).T, (avg_expr1, avg_expr2)
))

# add annos
pp.add_obs_annos(gadt1, avg_expr_add1, ignore_index=True)
pp.add_obs_annos(gadt2, avg_expr_add2, ignore_index=True)

''' plot cell type gene-profiles (plot all the cell types) on UMAP '''
ctypes1 = avg_expr1.columns.tolist()
ctypes2 = avg_expr2.columns.tolist()
sc.set_figure_params(fontsize=14)
cmap_expr = 'RdYlBu_r'
vmax = None
vmin = - 1.5
plkwds = dict(color_map=cmap_expr, vmax=vmax, vmin=vmin, ncols=5, )
sc.pl.umap(gadt1, color=ctypes1,
           #           edges=True, size=50,
           save=f'_exprAvgs-{dsn1}-all.png', **plkwds)
sc.pl.umap(gadt2, color=ctypes2,
           #           edges=True, size=50,
           save=f'_exprAvgs-{dsn2}-all.png', **plkwds)


