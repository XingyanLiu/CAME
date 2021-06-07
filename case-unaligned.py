# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:39:48 2021

@author: Xingyan Liu
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
subdir_res0 = f"{tiss}-{dsnames}{_time_tag}"

resdir = Path('./_case_res') / subdir_res0
figdir = resdir / 'figs'
sc.settings.figdir = figdir
CAME.check_dirs(figdir)

# In[]
''' loading data
'''
key_class1 = ['major_class', 'cell_ontology_class'][1]
key_class2 = key_class1

adata_raw1 = sc.read_h5ad(dir_formal / f'raw-{dsn1}.h5ad')
adata_raw2 = sc.read_h5ad(dir_formal / f'raw-{dsn2}.h5ad')
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

dpair, trainer, h_dict = pipeline.main_for_unaligned(
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
)

# out_cell = trainer.eval_current()['cell']
# probas_all = CAME.as_probabilities(out_cell)
# df_probs = pd.DataFrame(probas_all, columns = trainer.classes)

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

adt.write(resdir / 'adt_hidden_cell.h5ad')
gadt.write(resdir / 'adt_hidden_gene.h5ad')

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
ax.figure.show()

# In[]
'''===================== gene embeddings ====================='''
sc.set_figure_params(dpi_save=200)

sc.pp.neighbors(gadt, n_neighbors=15, metric='cosine', use_rep='X')
sc.tl.umap(gadt)
sc.pl.umap(gadt, color='dataset', )

''' joint gene module extraction '''
sc.tl.leiden(gadt, resolution=.8, key_added='module')
sc.pl.umap(gadt, color=['dataset', 'module'], ncols=1)

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
''' gene 3d map
'''

# In[]
''' ============ cell type gene-profiles on gene embeddings ============
'''
# averaged expressions
avg_expr1 = pp.group_mean_adata(adata_raw1, groupby=key_class1,
                                features=dpair.vnode_names1, use_raw=True)
avg_expr2 = pp.group_mean_adata(adata_raw2, groupby=key_class2,
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

# gadt1.write(resdir / 'adt_hidden_gene1.h5ad')
# gadt2.write(resdir / 'adt_hidden_gene2.h5ad')


# In[]
''' gene annotation on UMAP (top DEGs)
'''
fdir_gmap = resdir / 'gene_umap'
CAME.check_dirs(fdir_gmap)

adata1.obs[key_class1] = pd.Categorical(obs[key_class1][obs_ids1],
                                        categories=classes)
adata2.obs['predicted'] = pd.Categorical(obs['predicted'][obs_ids2],
                                         categories=classes)
pp.add_obs_annos(adata2, obs[classes].iloc[obs_ids2], ignore_index=True)

df_top1 = pp.compute_and_get_DEGs(adata1, key_class1, unique=False, )
df_top2 = pp.compute_and_get_DEGs(adata2, 'predicted',
                                  unique=False, )  # line 749
for _adt, sp, _df_top in zip([gadt1, gadt2],
                             (sp1, sp2),
                             (df_top1, df_top2),
                             ):
    for c in _df_top.columns[: 2]:
        text_ids = _df_top[c].head(10)

        nm = _adt.obs['dataset'][0]
        ftps = ['pdf', 'svg', 'png']
        ftp = ftps[1]
        ax = pl.umap_with_annotates(_adt, color=c, text_ids=text_ids,
                                    #                                      edges=True, size=30,
                                    title=f'{sp} {c}',
                                    index_col='name',
                                    fp=fdir_gmap / f'hexpr-{c}-{nm}.{ftp}',
                                    **plkwds)
        ax.figure

# In[]
''' =================== abstracted graph ====================  '''
norm_ov = ['max', 'zs', None][1]
cut_ov = 0.  # 5#(0.5, 2.5)
# norm_ov = 'zs'
# cut_ov = (0.5, 2.5)
ovby = ['expr', 'attn'][0]
groupby_var = 'module'
obs_labels1, obs_labels2 = adt.obs['celltype'][dpair.obs_ids1], \
                           adt.obs['celltype'][dpair.obs_ids2]
var_labels1, var_labels2 = gadt1.obs[groupby_var], gadt2.obs[groupby_var]

g = CAME.make_abstracted_graph(
    obs_labels1, obs_labels2,
    var_labels1, var_labels2,
    avg_expr1, avg_expr2,
    df_var_links,
    tags_obs=(f'{sp1} ', f'{sp2} '),
    tags_var=(f'{sp1} module ', f'{sp2} module '),
    cut_ov=cut_ov,
    norm_mtd_ov=norm_ov,
)

''' visualization '''
fp_abs = figdir / f'abstracted_graph-{groupby_var}-cut{cut_ov}-{norm_ov}-{ovby}.pdf'
ax = pl.plot_multipartite_graph(
    g, edge_scale=10,
    figsize=(9, 7.5), alpha=0.5, fp=fp_abs)  # nodelist=nodelist,

ax.figure
# unlabeled
ax = pl.plot_multipartite_graph(
    g, edge_scale=10, figsize=(9, 7.5), alpha=0.5,
    xscale=1.25,
    fp=figdir / f'abstracted_graph-nolabels.pdf',
    with_labels=False)  # nodelist=nodelist,

# In[]
human_tf = pd.read_csv(f'../resources/TF/fantomTFs-human.csv')['Symbol']
mouse_tf = pd.read_csv(f'../resources/TF/fantomTFs-mouse.csv')['Symbol']
TFdict = {'mouse': mouse_tf,
          'human': human_tf,
          }

# In[]
''' TF-target exploration '''

''' annotate TFs on gene UMAP '''
