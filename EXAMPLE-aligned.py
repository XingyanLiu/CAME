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
    # df_varmap_1v1 = None  # for within species

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
    batch_size = None  # None

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

    came_inputs, (adata1, adata2) = pipeline.preprocess_aligned(
        adatas,
        key_class=key_class1,
        df_varmap_1v1=df_varmap_1v1,
        use_scnets=use_scnets,
        ntop_deg=ntop_deg,
        node_source=node_source,
    )

    outputs = pipeline.main_for_aligned(
        **came_inputs,
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
    # sc.pl.umap(adt, color=['dataset', 'celltype'], ncols=1)

    ftype = ['.svg', ''][1]
    sc.pl.umap(adt, color='dataset', save=f'-dataset{ftype}')
    sc.pl.umap(adt, color='predicted', save=f'-ctype{ftype}')

    adt1, adt2 = pp.bisplit_adata(adt, key='dataset', left_groups=[dsn1])
    sc.pl.umap(adt1, color=['REF', 'predicted'], save=f'-sep1-{ftype}')
    sc.pl.umap(adt2, color=['celltype', 'predicted'], save=f'-sep2-{ftype}')

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
    """Check DEG (marker) expressions"""
    # check markers (not considering homologous mappings)
    degs = pp.compute_and_get_DEGs(adata1, key_class1, n=6)
    sc.pl.dotplot(adata1, degs, save='_ref', groupby=key_class1)
    logging.info(f"len(degs)={len(degs)}")

    degs2 = [g for g in degs if g in adata2.raw.var_names]

    sc.pl.dotplot(adata2, degs2, save='_que', groupby='predicted',
                  standard_scale='var')
    sc.pl.dotplot(adata1, degs2, save='_ref_union', groupby=key_class1,
                  standard_scale='var')
    # ==========================================================
    """ UMAP of gene embeddings """
    sc.pp.neighbors(gadt, n_neighbors=10, metric='cosine', use_rep='X')
    sc.tl.umap(gadt, )
    # joint gene module extraction
    sc.tl.leiden(gadt, resolution=1, key_added='module')
    sc.pl.umap(gadt, color='module', ncols=1, palette='tab20b',
               save=f'-gene_modules')

    """ Gene-expression-profiles (for each cell type) on gene UMAP """
    # averaged expressions
    avg_expr1 = pp.group_mean_adata(
        adata1, groupby=key_class1,
        features=dpair.var['name'], use_raw=True)
    avg_expr2 = pp.group_mean_adata(
        adata2, groupby='predicted',
        features=dpair.var['name'], use_raw=True)
    avg_expr1.columns = [f'{x}-1' for x in avg_expr1.columns]
    avg_expr2.columns = [f'{x}-2' for x in avg_expr2.columns]

    # z-scores across cell types
    avg_expr_add1, avg_expr_add2 = list(map(
        lambda x: pp.zscore(x.T).T, (avg_expr1, avg_expr2)
    ))
    # add annos
    pp.add_obs_annos(gadt, avg_expr_add1, ignore_index=True)
    pp.add_obs_annos(gadt, avg_expr_add2, ignore_index=True)

    gadt.write(resdir / 'adt_hidden_gene.h5ad')  # scanpy raise errors
    # gadt2.write(resdir / 'adt_hidden_gene2.h5ad')

    # ==================================================================
    # plot cell type gene-profiles (plot all the cell types) on UMAP
    sc.set_figure_params(fontsize=14)

    ctypes1 = list(avg_expr1.columns)
    ctypes2 = list(avg_expr2.columns)
    cmap_expr = 'RdYlBu_r'
    vmax = None
    vmin = - 1.5
    plkwds = dict(color_map=cmap_expr, vmax=vmax, vmin=vmin, ncols=4, )

    sc.pl.umap(gadt, color=ctypes1,
               # edges=True, size=50,
               save=f'_exprAvgs-{dsn1}-all.png', **plkwds)
    sc.pl.umap(gadt, color=ctypes2,
               # edges=True, size=50,
               save=f'_exprAvgs-{dsn2}-all.png', **plkwds)


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
