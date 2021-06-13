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
    dsnames = DATASET_PAIRS[-2]  # [::-1]
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
    params_model=dict(residual=False)
)
load_other_ckpt = not False
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


obs_ids1, obs_ids2 = dpair.obs_ids1, dpair.obs_ids2
obs = dpair.obs
classes = dpair.classes.copy()
if 'unknown' in classes:
    classes.remove('unknown')
    
# In[]
# ============== heatmap of predicted probabilities ==============
name_label = 'celltype'
cols_anno = ['celltype', 'predicted'][:]

out_cell = trainer.eval_current()['cell']

probas_all = CAME.as_probabilities(out_cell)
probas_all = CAME.model.detach2numpy(torch.sigmoid(out_cell))
#probas_all = np.apply_along_axis(lambda x: x / x.sum(), 1, probas_all,)
df_probs = pd.DataFrame(probas_all, columns=classes)

for i, _obs_ids in enumerate([obs_ids1, obs_ids2]):
    # df_lbs = obs[cols_anno][obs[key_class1] == 'unknown'].sort_values(cols_anno)
    df_lbs = obs[cols_anno].iloc[_obs_ids].sort_values(cols_anno)
    
    indices = CAME.subsample_each_group(df_lbs['celltype'], n_out=50, )
    # indices = df_lbs.index
    df_data = df_probs.loc[indices, :].copy()
    df_data = df_data[sorted(df_lbs['predicted'].unique())]  # .T
    lbs = df_lbs[name_label][indices]
    
    _ = pl.heatmap_probas(
        df_data.T, lbs, name_label='true label', 
        cmap_heat='RdBu_r',
        figsize=(5, 3.), fp=figdir / f'heatmap_probas-{i}.pdf'
    )
    
# In[]
# ======================= further analysis =======================
#trainer.model.rgcn.hidden_states
#h_dict = trainer.model.get_hidden_states()
h_dict_all = CAME.model.get_all_hidden_states(
        trainer.model, trainer.feat_dict, trainer.g
        )
h_dict = h_dict_all[-1]
adt = pp.make_adata(h_dict['cell'], obs=dpair.obs, assparse=False)
gadt = pp.make_adata(h_dict['gene'], obs=dpair.var, assparse=False)


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

# In[]
import umap
def umap_reduce0(
        X, 
        n_neighbors=15,
        metric='cosine',
        **kwargs
    ):
    reducer = umap.UMAP(n_neighbors=n_neighbors, metric=metric, **kwargs)
    return reducer.fit_transform(X)

def umap_reduce(
        X, 
        n_neighbors=15,
        metric='cosine',
        **kwargs
        ):
    _adt = pp.make_adata(X, assparse=False)
    sc.pp.neighbors(
            _adt,
            n_neighbors=n_neighbors,
            metric=metric,
            use_rep='X'
            )
    sc.tl.umap(_adt)
    return _adt.obsm['X_umap']
    

h_dict_all = CAME.model.get_all_hidden_states(
        trainer.model, trainer.feat_dict, trainer.g
        )
attn_mat = CAME.model.get_attentions(
        trainer.model, trainer.feat_dict, trainer.g, 
#        from_scratch=False
        )
#tmp = h_dict_all[0]['cell']
#tmp.shape
#umap_reduce(tmp)
# In[]
obs_umaps_all = [
        umap_reduce(h['cell']) for h in h_dict_all
        ]

for i, _x_umap in enumerate(obs_umaps_all):    
    adt.obsm[f'X_umap_layer{i}'] = _x_umap
    sc.pl.embedding(
            adt, color=['dataset', 'celltype'], basis=f'umap_layer{i}',
            show=True, 
            save=f'_cell_layer_{i}'
            )
    
# In[]
''' separately cell UMAP '''
adt1, adt2 = pp.bisplit_adata(adt, 'dataset', left_groups=[dsn1])

for _adt, _obs_ids, _tag in zip([adt1, adt2], 
                          [obs_ids1, obs_ids2],
                          ['ref', 'que']
                          ):
    obs_umaps_all = [
            umap_reduce(h['cell'][_obs_ids]) for h in h_dict_all
            ]
    
    for i, _x_umap in enumerate(obs_umaps_all):    
        _adt.obsm[f'X_umap_layer{i}'] = _x_umap
        sc.pl.embedding(
                _adt, color='celltype', 
                basis=f'umap_layer{i}',
                show=True, 
                save=f'_{_tag}_cell_layer_{i}'
                )

 
# In[]
''' combined gene UMAP at each layer'''

sc.pp.neighbors(gadt, n_neighbors=15, metric='cosine', use_rep='X')
sc.tl.leiden(gadt, resolution=0.8, key_added='module')

# In[]

var_umaps_all = [
        umap_reduce(h['gene']) for h in h_dict_all
        ]
for i, _x_umap in enumerate(var_umaps_all):    
    gadt.obsm[f'X_umap_layer{i}'] = _x_umap
    sc.pl.embedding(
            gadt, color=['dataset', 'module'], 
            basis=f'layer{i}_umap',
            show=True, 
            save=f'_gene'
            )
    

# In[]
''' independent gene UMAP at each layer'''
var_ids1, var_ids2 = dpair.get_vnode_ids(0), dpair.get_vnode_ids(1)
gadt1, gadt2 = pp.bisplit_adata(gadt, 'dataset', left_groups=[dsn1])

for _adt, _var_ids, _tag in zip([gadt1, gadt2], 
                          [var_ids1, var_ids2],
                          ['ref', 'que']
                          ):
    obs_umaps_all = [
            umap_reduce(h['gene'][_var_ids]) for h in h_dict_all
            ]
    
    for i, _x_umap in enumerate(obs_umaps_all):    
        _adt.obsm[f'X_umap_layer{i}'] = _x_umap
        sc.pl.embedding(
                _adt, color='module', 
                basis=f'umap_layer{i}',
                show=True, 
                save=f'_{_tag}_gene_layer_{i}'
                )
    
# In[]
''' attention matrix'''
attn1 = attn_mat[obs_ids1, :][:, var_ids1]
attn2 = attn_mat[obs_ids2, :][:, var_ids2]
    
adta1 = pp.make_adata(attn1, obs=adt1.obs)
adta2 = pp.make_adata(attn2, obs=adt2.obs)
adta1.var_names = dpair.vnode_names1
adta2.var_names = dpair.vnode_names2


sc.tl.pca(adta1, n_comps=50)
sc.pp.neighbors(adta1, n_neighbors=15, metric='cosine')
sc.tl.umap(adta1)
sc.pl.umap(adta1, color='celltype')
sc.pl.pca(adta1, color='celltype')


sc.tl.pca(adta2, n_comps=50)
sc.pp.neighbors(adta2, n_neighbors=15, metric='cosine')
sc.tl.umap(adta2)
sc.pl.umap(adta2, color='celltype')
sc.pl.pca(adta2, color='celltype')


adta1_, adta2_ = pp.align_adata_vars(
        adta1, adta2, df_varmap_1v1, unify_names=True)
adta_ = pp.merge_adatas([adta1_, adta2_], union=False)
pp.add_obs_annos(adta_, adt.obs)

sc.tl.pca(adta_, n_comps=50)
sc.pp.neighbors(adta_, n_neighbors=15, metric='cosine')
sc.tl.umap(adta_)
sc.pl.umap(adta_, color=['dataset', 'celltype'])
sc.pl.pca(adta_, color=['dataset', 'celltype'])
