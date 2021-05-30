# -*- coding: UTF-8 -*-
"""
@author: Xingyan Liu
@file: pipeline_supervised.py
@time: 2021-05-25
"""

import os
from pathlib import Path
from typing import Sequence, Union, Mapping, List, Optional  # , Callable
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import scanpy as sc
from scipy import sparse
from scipy.special import softmax

import networkx as nx
import torch
import logging

from . import (
    save_pickle,
    make_nowtime_tag,
    write_info,
    as_probabilities,
    predict_from_logits,
    subsample_each_group,
)
from .PARAMETERS import (
    get_model_params,
    get_loss_params,
    get_preprocess_params
)
from . import pp as utp
from . import pl as uplt
from .utils import base, evaluation
from . import (
    CGGCNet, datapair_from_adatas,
    CGCNet, aligned_datapair_from_adatas
)
from .utils.train_for_both import prepare4train, Trainer, seed_everything

PARAMS_MODEL = get_model_params()
PARAMS_PRE = get_preprocess_params()
PARAMS_LOSS = get_loss_params()

def main_for_unaligned(
        adatas: sc.AnnData,
        vars_use: Sequence[Sequence],
        vars_as_nodes: Sequence[Sequence],
        df_varmap: pd.DataFrame,
        df_varmap_1v1: Optional[pd.DataFrame] = None,
        scnets: Optional[Sequence[sparse.spmatrix]] = None,
        dataset_names: Sequence[str] = ('reference', 'query'),
        key_class1: str = 'cell_ontology_class',
        key_class2: Optional[str] = None,
        do_normalize: bool = False,
        n_epochs: int = 350,
        resdir: Union[Path, str] = None,
        tag_data: Optional[str] = None,  # for autometically deciding `resdir` for results saving
        params_pre: dict = PARAMS_PRE,
        params_model: dict = PARAMS_MODEL,
        params_lossfunc: dict = PARAMS_LOSS,
        check_umap: bool = False,  # TODO
        n_pass: int = 100,
):
    if resdir is None:
        tag_time = base.make_nowtime_tag()
        tag_data = dataset_names if tag_data is None else tag_data
        resdir = Path(f'{tag_data}-{tag_time}')
    else:
        resdir = Path(resdir)

    figdir = resdir / 'figs'
    utp.check_dirs(figdir)
    sc.settings.figdir = figdir
    utp.check_dirs(resdir)
    # keys for training process
    if key_class2 is None:
        if key_class1 in adatas[1].obs.columns:
            key_class2 = key_class1
            keys = [key_class1, key_class2]
        else:
            key_class2 = 'clust_lbs'
            keys = [key_class1, None]
    else:
        keys = [key_class1, key_class2]
    keys_compare = [key_class1, key_class2]

    if do_normalize:
        adatas = list(map(
            lambda a: utp.normalize_default(a, force_return=True),
            adatas
        ))
    logging.info('preparing DataPair object...')
    dpair = datapair_from_adatas(
        adatas,
        vars_use=vars_use,
        df_varmap=df_varmap,
        df_varmap_1v1=df_varmap_1v1,
        oo_adjs=scnets,
        vars_as_nodes=vars_as_nodes,
        union_node_feats='auto',
        dataset_names=dataset_names,
    )

    ENV_VARs = prepare4train(dpair, key_class=keys, )

    logging.info(ENV_VARs.keys())
    G = ENV_VARs['G']
    classes1, classes2 = ENV_VARs['classes']
    n_classes1, n_classes2 = len(classes1), len(classes2)

    params_model.update(
        in_dim_dict={'cell': dpair.n_feats, 'gene': 0},
        out_dim=n_classes1 + n_classes2,
        layernorm_ntypes=G.ntypes,
    )

    # model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CGGCNet(G, **params_model)

    ''' Training '''
    trainer = Trainer(model=model, g=G, dir_main=resdir, **ENV_VARs)
    trainer.train(n_epochs=n_epochs,
                  params_lossfunc=params_lossfunc,
                  n_pass=n_pass, )
    trainer.save_model_weights()
    trainer.load_model_weights()  # 127)
    # trainer.load_model_weights(trainer._cur_epoch)
    test_acc1 = trainer.test_acc1[trainer._cur_epoch_adopted]
    test_acc2 = trainer.test_acc2[trainer._cur_epoch_adopted]

    '''========================== record results ========================
    '''
    trainer.write_train_logs()

    write_info(resdir / 'info.txt',
               current_performance=trainer._cur_log,
               params_model=params_model,
               graph=G,
               model=model,
               )
    # trainer.plot_cluster_index(fp=figdir / 'cluster_index.png')

    ''' ======================== Gather results ======================
    '''
    obs_ids1, obs_ids2 = dpair.get_obs_ids(0, False), dpair.get_obs_ids(1, False)
    out_cell = trainer.eval_current()['cell']
    out_cell1 = out_cell[:, : n_classes1]
    out_cell2 = out_cell[:, n_classes1:]

    probas_all1 = as_probabilities(out_cell1)
    probas_all2 = as_probabilities(out_cell2)
    cl_preds1 = predict_from_logits(probas_all1, classes=classes1)
    cl_preds2 = predict_from_logits(probas_all2, classes=classes2)
    labels_cat = dpair.get_obs_labels(keys, asint=False)
    obs = pd.DataFrame(
        {key_class1: labels_cat,  # true labels with `unknown` for unseen classes in query data
         'celltype': dpair.get_obs_anno(keys_compare),  # labels for comparison
         'predicted1': cl_preds1,
         'predicted2': cl_preds2,
         'max_probs1': np.max(probas_all1, axis=1),
         'max_probs2': np.max(probas_all2, axis=1),
         })
    df_probs1 = pd.DataFrame(probas_all1, columns=classes1)
    df_probs2 = pd.DataFrame(probas_all2, columns=classes2)
    df_probs1.to_csv(resdir / f'probabilities_1.csv')
    df_probs2.to_csv(resdir / f'probabilities_2.csv')
    dpair.set_common_obs_annos(obs)
    # TODO: `df_probs1` and `df_probs1` may have conflicts
    dpair.set_common_obs_annos(df_probs1, ignore_index=True)
    dpair.obs.to_csv(resdir / 'obs.csv')
    save_pickle(dpair, resdir / 'dpair.pickle')
    # adding labels, predicted probabilities
    #    adata1.obs[key_class1] = pd.Categorical(obs[key_class1][obs_ids1], categories=classes)
    #    adata2.obs['predicted'] = pd.Categorical(obs['predicted'][obs_ids2], categories=classes)
    #    utp.add_obs_annos(adata2, df_probs.iloc[obs_ids2], ignore_index=True)

    # hidden states are stored in sc.AnnData to facilitated downstream analysis
    h_dict = model.get_hidden_states()  # trainer.feat_dict, trainer.g)
    adt = utp.make_adata(h_dict['cell'], obs=dpair.obs, assparse=False)
    #    gadt = utp.make_adata(h_dict['gene'], obs = adpair.var, assparse=False)

    ### group counts statistics (optinal)
    gcnt = utp.group_value_counts(dpair.obs, 'celltype', group_by='dataset')
    logging.info(str(gcnt))
    gcnt.to_csv(resdir / 'group_counts.csv')

    # ============= confusion matrix OR alluvial plot ==============
    sc.set_figure_params(fontsize=10)
    # TODO: cross-table for both
    lblist_y = [labels_cat[obs_ids1], labels_cat[obs_ids2]]
    lblist_x = [cl_preds1[obs_ids1], cl_preds1[obs_ids2]]

    uplt.plot_confus_multi_mats(
        lblist_y,
        lblist_x,
        classes_on=classes1,
        fname=figdir / f'confusion_matrix(acc{test_acc1:.1%}).png',
    )
    # ============== heatmap of predicted probabilities ==============
    name_label = 'celltype'
    cols_anno = ['celltype', 'predicted1'][:]

    # df_lbs = obs[cols_anno][obs[key_class1] == 'unknown'].sort_values(cols_anno)
    df_lbs = obs[cols_anno].iloc[obs_ids2].sort_values(cols_anno)

    indices = subsample_each_group(df_lbs['celltype'], n_out=50, )
    # indices = df_lbs.index
    df_data1 = df_probs1.loc[indices, :].copy()
    df_data1 = df_data1[sorted(df_lbs['predicted1'].unique())]  # .T
    lbs = df_lbs[name_label][indices]
    # TODO: heatmap_probas for both
    _ = uplt.heatmap_probas(df_data1.T, lbs, name_label='true label',
                            figsize=(5, 3.),
                            fp=figdir / f'heatmap_probas.pdf'
                            )
    return dpair, trainer, h_dict


def preprocess_unaligned(
        adatas,
        key_class: str,
        use_scnets: bool = True,
        n_pcs: int = 30,
        nneigh_scnet: int = 5,
        nneigh_clust: int = 20,
        ntop_deg: int = 50,
):
    logging.info('================ preprocessing ===============')
    params_preproc = dict(
        target_sum=None,
        n_top_genes=2000,
        n_pcs=n_pcs,
        nneigh=nneigh_scnet,
    )
    # NOTE: using the median total-counts as the scale factor (better than fixed number)
    adata1 = utp.quick_preprocess(adatas[0], **params_preproc)
    adata2 = utp.quick_preprocess(adatas[1], **params_preproc)

    # the single-cell network
    if use_scnets:
        scnets = [utp.get_scnet(adata1), utp.get_scnet(adata2)]
    else:
        scnets = None
    # get HVGs
    hvgs1, hvgs2 = utp.get_hvgs(adata1), utp.get_hvgs(adata2)

    # cluster labels
    key_clust = 'clust_lbs'
    clust_lbs2 = utp.get_leiden_labels(
        adata2, force_redo=True,
        nneigh=nneigh_clust,
        neighbors_key='clust',
        key_added=key_clust,
        copy=False
    )
    adatas[1].obs[key_clust] = clust_lbs2

    params_deg = dict(n=ntop_deg, force_redo=False,
                      inplace=True, do_normalize=False)
    ### need to be normalized first
    degs1 = utp.compute_and_get_DEGs(
        adata1, key_class, **params_deg)
    degs2 = utp.compute_and_get_DEGs(
        adata2, key_clust, **params_deg)
    ###
    vars_use = [degs1, degs2]
    vars_as_nodes = [np.unique(np.hstack([hvgs1, degs1])),
                     np.unique(np.hstack([hvgs2, degs2]))]

    dct = dict(
        adatas=adatas,
        vars_use=vars_use,
        vars_as_nodes=vars_as_nodes,
        scnets=scnets,
    )
    return dct, (adata1, adata2)


def __test2_sup__(n_epochs: int = 5):
    seed_everything()
    datadir = Path(os.path.abspath(__file__)).parent / 'sample_data'
    sp1, sp2 = ('human', 'mouse')
    dsnames = ('Baron_human', 'Baron_mouse')

    df_varmap_1v1 = pd.read_csv(datadir / f'gene_matches_1v1_{sp1}2{sp2}.csv', )
    df_varmap = pd.read_csv(datadir / f'gene_matches_{sp1}2{sp2}.csv', )

    dsn1, dsn2 = dsnames
    adata_raw1 = sc.read_h5ad(datadir / f'raw-{dsn1}.h5ad')
    adata_raw2 = sc.read_h5ad(datadir / f'raw-{dsn2}.h5ad')
    adatas = [adata_raw1, adata_raw2]

    key_class = 'cell_ontology_class'
    time_tag = make_nowtime_tag()
    resdir = Path('_temp') / f'{dsnames}-{time_tag}'

    came_inputs, (adata1, adata2) = preprocess_unaligned(
        adatas,
        key_class=key_class,
    )

    dpair, trainer, _ = main_for_unaligned(
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

    del dpair, trainer, _
    torch.cuda.empty_cache()
    logging.debug('memory cleared\n')
    print('Test passed for UN-ALIGNED!')

