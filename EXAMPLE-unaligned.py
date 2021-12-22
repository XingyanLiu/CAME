# -*- coding: UTF-8 -*-
"""
@Author: Xingyan Liu
@CreateDate: 2021-12-08
@File: EXAMPLE-aligned.py
@Project: CAME
"""
import os
import sys
from pathlib import Path
from typing import Union, Optional, Sequence, Mapping
import time
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import scanpy as sc
from scipy import sparse
import networkx as nx
import torch

ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))
import came
from came import pipeline, pp, pl

EXAMPLE_DATADIR = Path('E:/lxy_pro/004/CAME/came/sample_data')


def load_example_adatas():
    fp_adt1 = EXAMPLE_DATADIR / 'raw-Baron_human.h5ad'
    fp_adt2 = EXAMPLE_DATADIR / 'raw-Baron_mouse.h5ad'
    adata_raw1 = sc.read_h5ad(fp_adt1)
    adata_raw2 = sc.read_h5ad(fp_adt2)
    return adata_raw1, adata_raw2


def load_adatas(fp_adt1, fp_adt2):
    adata_raw1 = sc.read_h5ad(fp_adt1)
    adata_raw2 = sc.read_h5ad(fp_adt2)
    return adata_raw1, adata_raw2


def load_varmap(fp_varmap=None):
    if fp_varmap is None:
        fp_varmap = EXAMPLE_DATADIR / 'gene_matches_human2mouse.csv'
    return pd.read_csv(fp_varmap)


def make_var_map(
    df_varmap_1v1,
    icol_index=0, icol_val=1,
    col_index=None, col_val=None
):
    if col_index is None:
        col_index = df_varmap_1v1.columns[icol_index]
    var_map = df_varmap_1v1.set_index(col_index, drop=False)
    if col_val is None:
        var_map = var_map.iloc[:, icol_val].to_dict()
    else:
        var_map = var_map[icol_val].to_dict()
    return var_map


def filter_non1v1homo(genes, var_map, universe):
    # universe = adata2.raw.var_names
    degs1, degs2 = [], []
    for g1 in genes:
        g2 = var_map.get(g1, None)
        if g2 in universe:
            degs1.append(g1)
            degs2.append(g2)
    return degs1, degs2


def main():
    # ======= basic settings =======
    key_class1 = 'cell_ontology_class'
    key_class2 = key_class1
    dsnames = (dsn1, dsn2) = ('Baron_human', 'Baron_mouse')

    # ============ load example data ================
    adata_raw1, adata_raw2 = load_example_adatas()
    df_varmap = load_varmap()
    df_varmap_1v1 = pp.take_1v1_matches(df_varmap)

    # ================ load data ====================
    # fp_adt1, fp_adt2 = '.h5ad', '.h5ad'
    # fp_varmap = '.csv'
    # adata_raw1, adata_raw2 = load_adatas(fp_adt1, fp_adt2)
    # df_varmap = load_varmap(fp_varmap)
    # df_varmap_1v1 = pp.take_1v1_matches(df_varmap)

    # =================== optional ==================
    # adata_raw1.obs_names_make_unique()
    # adata_raw2.obs_names_make_unique()
    # adata_raw1 = pp.remove_adata_groups(
    #     adata_raw1, key=key_class_major, group_names=['filtered'])
    # ===============================================

    adatas = [adata_raw1, adata_raw2]
    # ============== result directory ===============
    time_tag = came.make_nowtime_tag()
    resdir = ROOT / 'came_res' / f'{dsnames}-{time_tag}'
    figdir = resdir / 'figs'
    came.check_dirs(figdir)
    sc.settings.figdir = figdir
    sc.set_figure_params(dpi_save=200)

    # ================= settings of CAME =================
    # the training batch size
    # (when GPU memory is limited, recommanded is 1024 or more)
    batch_size = 1024  # None

    # the numer of training epochs
    n_epochs = 350

    # the number of epochs to skip for checkpoint backup
    n_pass = 80

    if batch_size:
        _n_batch = ((adatas[0].shape[0] + adatas[1].shape[0]) // batch_size + 1)
        n_epochs = (n_epochs + 150) // _n_batch
        n_pass = n_pass // _n_batch

    # whether to use the single-cell network
    use_scnets = True

    # node genes. Using DEGs and HVGs by default
    node_source = 'deg,hvg'  # 'deg,hvg'
    ntop_deg = 50
    # ===============================================

    came_inputs, (adata1, adata2) = pipeline.preprocess_unaligned(
        adatas,
        key_class=key_class1,
        use_scnets=use_scnets,
        ntop_deg=ntop_deg,
        node_source=node_source,
    )

    outputs = pipeline.main_for_unaligned(
        **came_inputs,
        df_varmap=df_varmap,
        df_varmap_1v1=df_varmap_1v1,
        dataset_names=dsnames,
        key_class1=key_class1,
        key_class2=key_class2,
        do_normalize=True,
        n_epochs=n_epochs,
        resdir=resdir,
        n_pass=n_pass,
        batch_size=batch_size,
        plot_results=True,
    )
    dpair = outputs['dpair']
    trainer = outputs['trainer']
    h_dict = outputs['h_dict']
    out_cell = outputs['out_cell']
    predictor = outputs['predictor']

    obs_ids1, obs_ids2 = dpair.obs_ids1, dpair.obs_ids2
    obs = dpair.obs
    classes = predictor.classes

    # ==========================================================
    """ UMAP of cell embeddings """
    adt = pp.make_adata(h_dict['cell'], obs=obs, assparse=False,
                        ignore_index=True)
    gadt = pp.make_adata(h_dict['gene'], obs=dpair.var, assparse=False,
                         ignore_index=True)

    sc.pp.neighbors(adt, n_neighbors=15, metric='cosine', use_rep='X')
    sc.tl.umap(adt)

    ftype = ['.svg', ''][1]
    sc.pl.umap(adt, color='dataset', save=f'-dataset{ftype}')
    sc.pl.umap(adt, color='predicted', save=f'-ctype{ftype}')

    adt1, adt2 = pp.bisplit_adata(adt, key='dataset', left_groups=[dsn1])
    sc.pl.umap(adt1, color=['REF', 'predicted'], save=f'-sep1-{ftype}')
    sc.pl.umap(adt2, color=['celltype', 'predicted'], save=f'-sep2-{ftype}')

    # set and order classes
    adata1.obs[key_class1] = pd.Categorical(
        adata1.obs[key_class1], categories=classes)
    adata2.obs['predicted'] = pd.Categorical(
        adt2.obs['predicted'].values, categories=classes)

    # store UMAP coordinates
    obs_umap = adt.obsm['X_umap']
    obs['UMAP1'] = obs_umap[:, 0]
    obs['UMAP2'] = obs_umap[:, 1]
    obs.to_csv(resdir / 'obs.csv')
    adt.write(resdir / 'adt_hidden_cell.h5ad')

    # ==============================================================
    """Check marker expressions"""
    # check markers (not considering homologous mappings)
    degs10 = pp.compute_and_get_DEGs(adata1, key_class1, n=6)
    logging.info(f"len(degs)={len(degs10)}")
    sc.pl.dotplot(adata1, degs10, save='_ref', groupby=key_class1)

    degs1, degs2 = filter_non1v1homo(
        degs10, make_var_map(df_varmap_1v1), universe=adata2.raw.var_names)
    sc.pl.dotplot(adata1, degs1, save='_ref_common1', groupby=key_class1,
                  standard_scale='var')
    sc.pl.dotplot(adata2, degs2, save='_que_common1', groupby='predicted',
                  standard_scale='var', color_map='Blues')

    """Check marker expressions (reversed, DEGs from query data) """
    groupby_de2 = 'predicted'
    degs20 = pp.compute_and_get_DEGs(adata2, groupby_de2, n=6)
    sc.pl.dotplot(adata2, degs20, save='_que', groupby=groupby_de2, color_map='Blues')
    degs2, degs1 = filter_non1v1homo(
        degs20, make_var_map(df_varmap_1v1, icol_index=1, icol_val=0),
        universe=adata1.raw.var_names)
    sc.pl.dotplot(adata1, degs1, save='_ref_common2', groupby=key_class1,
                  standard_scale='var')
    sc.pl.dotplot(adata2, degs2, save='_que_common2', groupby=groupby_de2,
                  standard_scale='var', color_map='Blues')

    # ==========================================================
    """ UMAP of gene embeddings """
    sc.pp.neighbors(gadt, n_neighbors=10, metric='cosine', use_rep='X')
    sc.tl.umap(gadt, )
    sc.pl.umap(gadt, color='dataset', save='-genes_by_species')
    # joint gene module extraction
    sc.tl.leiden(gadt, resolution=1, key_added='module')
    sc.pl.umap(gadt, color='module', ncols=1, palette='tab20b',
               save=f'-gene_modules')

    # link-weights between homologous gene pairs
    df_var_links = came.weight_linked_vars(
        gadt.X, dpair._vv_adj, names=dpair.get_vnode_names(),
        matric='cosine', index_names=dsnames,
    )

    """ Gene-expression-profiles (for each cell type) on gene UMAP """
    # gadt.obs_names = gadt.obs_names.astype(str)
    gadt1, gadt2 = pp.bisplit_adata(gadt, 'dataset', dsnames[0],
                                    reset_index_by='name')
    # averaged expressions
    avg_expr1 = pp.group_mean_adata(
        adata1, groupby=key_class1, features=dpair.vnode_names1, use_raw=True)
    avg_expr2 = pp.group_mean_adata(
        adata2, groupby=key_class2, features=dpair.vnode_names2, use_raw=True)

    # z-scores across cell types
    avg_expr_add1, avg_expr_add2 = list(map(
        lambda x: pp.zscore(x.T).T, (avg_expr1, avg_expr2)
    ))

    # add annos
    pp.add_obs_annos(gadt1, avg_expr_add1, ignore_index=True)
    pp.add_obs_annos(gadt2, avg_expr_add2, ignore_index=True)

    gadt1.write(resdir / 'adt_hidden_gene1.h5ad')
    gadt2.write(resdir / 'adt_hidden_gene2.h5ad')

    # ==================================================================
    # plot cell type gene-profiles (plot all the cell types) on UMAP
    sc.set_figure_params(fontsize=14)

    ctypes1 = avg_expr1.columns.tolist()
    ctypes2 = avg_expr2.columns.tolist()
    cmap_expr = 'RdYlBu_r'
    vmax = None
    vmin = - 1.5
    plkwds = dict(color_map=cmap_expr, vmax=vmax, vmin=vmin, ncols=5, )

    sc.pl.umap(gadt1, color=ctypes1,
               # edges=True, size=50,
               save=f'_exprAvgs-{dsn1}-all.png', **plkwds)
    sc.pl.umap(gadt2, color=ctypes2,
               # edges=True, size=50,
               save=f'_exprAvgs-{dsn2}-all.png', **plkwds)
    # ==================================================================
    """2.4 Abstracted graph"""
    norm_ov = ['max', 'zs', None][1]
    cut_ov = 0.  # 5#(0.5, 2.5)
    # norm_ov = 'zs'
    # cut_ov = (0.5, 2.5)
    ovby = ['expr', 'attn'][0]
    groupby_var = 'module'
    obs_labels1, obs_labels2 = adt.obs['celltype'][dpair.obs_ids1], \
                               adt.obs['celltype'][dpair.obs_ids2]
    var_labels1, var_labels2 = gadt1.obs[groupby_var], gadt2.obs[groupby_var]

    sp1, sp2 = 'human', 'mouse'
    g = came.make_abstracted_graph(
        obs_labels1, obs_labels2,
        var_labels1, var_labels2,
        avg_expr1, avg_expr2,
        df_var_links,
        tags_obs=(f'{sp1} ', f'{sp2} '),
        tags_var=(f'{sp1} module ', f'{sp2} module '),
        cut_ov=cut_ov,
        norm_mtd_ov=norm_ov,
    )
    # visualization
    fp_abs = figdir / f'abstracted_graph-{groupby_var}-cut{cut_ov}-{norm_ov}-{ovby}.pdf'
    ax = pl.plot_multipartite_graph(
        g, edge_scale=10, with_labels=True,
        figsize=(9, 7.5), alpha=0.5, fp=fp_abs)  # nodelist=nodelist,

    came.save_pickle(g, resdir / 'abs_graph.pickle')


def __test__():
    main()


if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('agg')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')
    t = time.time()

    __test__()

    print('Done running file: {}\nTime: {}'.format(
        os.path.abspath(__file__), time.time() - t,
    ))
