# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:11:08 2021

@author: Xingyan Liu
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
import torch
import logging
from . import (
    pp, pl,
    save_pickle,
    detach2numpy,
    save_json_dict,
    make_nowtime_tag,
    write_info,
    as_probabilities,
    predict_from_logits,
    subsample_each_group,
    SUBDIR_MODEL,
)
from .PARAMETERS import (
    get_model_params,
    get_loss_params,
    get_preprocess_params
)
from . import (
    Predictor,
    CGGCNet, datapair_from_adatas,
    CGCNet, aligned_datapair_from_adatas
)
from .utils.train import prepare4train, Trainer, seed_everything

PARAMS_MODEL = get_model_params()
PARAMS_PRE = get_preprocess_params()
PARAMS_LOSS = get_loss_params()

# In[] Main Functions


def main_for_aligned(
        adatas: sc.AnnData,
        vars_feat: Sequence,
        vars_as_nodes: Optional[Sequence] = None,
        scnets: Optional[Sequence[sparse.spmatrix]] = None,
        dataset_names: Sequence[str] = ('reference', 'query'),
        key_class1: str = 'cell_ontology_class',
        key_class2: Optional[str] = None,
        do_normalize: bool = False,
        n_epochs: int = 350,
        resdir: Union[Path, str] = None,
        tag_data: Optional[str] = None,  # for autometically deciding `resdir` for results saving
        params_model: dict = {},
        params_lossfunc: dict = {},
        n_pass: int = 100,
        batch_size: Optional[int] = None,
        plot_results: bool = True,
        norm_target_sum: Optional[float] = 1e4,
):
    if resdir is None:
        tag_time = make_nowtime_tag()
        tag_data = dataset_names if tag_data is None else tag_data
        resdir = Path(f'{tag_data}-{tag_time}')
    else:
        resdir = Path(resdir)

    figdir = resdir / 'figs'
    pp.check_dirs(figdir)
    sc.settings.figdir = figdir
    pp.check_dirs(resdir)
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
            lambda a: pp.normalize_default(
                a, target_sum=norm_target_sum, force_return=True), 
            adatas))

    logging.info('Step 1: preparing DataPair object...')
    adpair = aligned_datapair_from_adatas(
        adatas,
        vars_feat=vars_feat,
        vars_as_nodes=vars_as_nodes,
        oo_adjs=scnets,
        dataset_names=dataset_names,
    )

    ENV_VARs = prepare4train(adpair, key_class=keys, )

    logging.debug(ENV_VARs.keys())
    G = ENV_VARs['G']
    classes0 = ENV_VARs['classes']
    classes = classes0[:-1] if 'unknown' in classes0 else classes0
    n_classes = len(classes)
    params_model = get_model_params(**params_model)
    params_model.update(
        g_or_canonical_etypes=G.canonical_etypes,
        in_dim_dict={'cell': adpair.n_feats, 'gene': 0},
        out_dim=n_classes,
        layernorm_ntypes=G.ntypes,
    )
    save_json_dict(params_model, resdir / 'model_params.json')

    # TODO: save model parameters, json file (whether eval?)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CGCNet(**params_model)

    params_lossfunc = get_loss_params(**params_lossfunc)
    trainer = Trainer(model=model, g=G, dir_main=resdir, **ENV_VARs)

    if batch_size is not None:
        trainer.train_minibatch(
            n_epochs=n_epochs,
            params_lossfunc=params_lossfunc,
            batch_size=batch_size,
            n_pass=n_pass, )
    else:
        trainer.train(n_epochs=n_epochs,
                      params_lossfunc=params_lossfunc,
                      n_pass=n_pass, )
    trainer.save_model_weights()
    # ========================== record results ========================
    trainer.write_train_logs()

    write_info(resdir / 'info.txt',
               current_performance=trainer._cur_log,
               params_model=params_model,
               graph=G,
               model=model,
               )
    trainer.plot_cluster_index(fp=figdir / 'cluster_index.png')

    # ======================== Gather results ======================
    obs_ids1 = adpair.get_obs_ids(0, False)
    obs_ids2 = adpair.get_obs_ids(1, False)
    obs, df_probs, h_dict, predictor = gather_came_results(
        adpair,
        trainer,
        classes=classes,
        keys=keys,
        keys_compare=keys_compare,
        resdir=resdir,
        checkpoint='best',
        batch_size=batch_size,
    )
    test_acc = trainer.test_acc[trainer._cur_epoch_adopted]

    # ============= confusion matrix & heatmap plot ==============
    if plot_results:
        labels_cat = obs[keys[0]] if key_class2 != 'clust_lbs' else obs['celltype']
        cl_preds = obs['predicted']
        sc.set_figure_params(fontsize=10)
    
        # confusion matrix OR alluvial plot
        sc.set_figure_params(fontsize=10)
        ax, contmat = pl.plot_contingency_mat(
            labels_cat[obs_ids2], cl_preds[obs_ids2], norm_axis=1,
            fp=figdir / f'contingency_matrix(acc{test_acc:.1%}).png',
        )
        pl.plot_confus_mat(
            labels_cat[obs_ids1], cl_preds[obs_ids1], classes_on=classes,
            fp=figdir / f'contingency_matrix-train.png',
        )

        # heatmap of predicted probabilities
        gs = pl.wrapper_heatmap_scores(
            df_probs.iloc[obs_ids2], obs.iloc[obs_ids2], ignore_index=True,
            col_label='celltype', col_pred='predicted',
            n_subsample=50,
            cmap_heat='magma_r', #if prob_func == 'softmax' else 'RdBu_r'
            fp=figdir / f'heatmap_probas.pdf'
        )

    return adpair, trainer, h_dict, predictor, ENV_VARs


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
        params_model: dict = {},
        params_lossfunc: dict = {},
        check_umap: bool = False,  # TODO
        n_pass: int = 100,
        plot_results: bool = True,
        batch_size: Optional[int] = None,
):
    if resdir is None:
        tag_time = make_nowtime_tag()
        tag_data = dataset_names if tag_data is None else tag_data
        resdir = Path(f'{tag_data}-{tag_time}')
    else:
        resdir = Path(resdir)

    figdir = resdir / 'figs'
    pp.check_dirs(figdir)
    sc.settings.figdir = figdir
    pp.check_dirs(resdir)
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
            lambda a: pp.normalize_default(a, force_return=True),
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
    classes0 = ENV_VARs['classes']
    classes = classes0[:-1] if 'unknown' in classes0 else classes0
    n_classes = len(classes)
    params_model = get_model_params(**params_model)
    params_model.update(
        g_or_canonical_etypes=G.canonical_etypes,
        in_dim_dict={'cell': dpair.n_feats, 'gene': 0},
        out_dim=n_classes,
        layernorm_ntypes=G.ntypes,
    )
    save_json_dict(params_model, resdir / 'model_params.json')

    # TODO: save model parameters, json file (whether eval?)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CGGCNet(**params_model)

    # training
    params_lossfunc = get_loss_params(**params_lossfunc)
    trainer = Trainer(model=model, g=G, dir_main=resdir, **ENV_VARs)
    if batch_size is not None:
        trainer.train_minibatch(
            n_epochs=n_epochs,
            params_lossfunc=params_lossfunc,
            batch_size=batch_size,
            n_pass=n_pass, )
    else:
        trainer = Trainer(model=model, g=G, dir_main=resdir, **ENV_VARs)
        trainer.train(n_epochs=n_epochs,
                      params_lossfunc=params_lossfunc,
                      n_pass=n_pass, )
    trainer.save_model_weights()
    # ========================== record results ========================
    trainer.write_train_logs()
    write_info(resdir / 'info.txt',
               current_performance=trainer._cur_log,
               params_model=params_model,
               graph=G,
               model=model,
               )
    trainer.plot_cluster_index(fp=figdir / 'cluster_index.png')

    # ======================== Gather results ======================
    obs_ids1 = dpair.get_obs_ids(0, False)
    obs_ids2 = dpair.get_obs_ids(1, False)
    obs, df_probs, h_dict, predictor = gather_came_results(
        dpair,
        trainer,
        classes=classes,
        keys=keys,
        keys_compare=keys_compare,
        resdir=resdir,
        checkpoint='best',
        batch_size=batch_size,
    )
    test_acc = trainer.test_acc[trainer._cur_epoch_adopted]

    if plot_results:
        labels_cat = obs[keys[0]] if key_class2 != 'clust_lbs' else obs['celltype']
        cl_preds = obs['predicted']

        # confusion matrix OR alluvial plot
        sc.set_figure_params(fontsize=10)
        ax, contmat = pl.plot_contingency_mat(
            labels_cat[obs_ids2], cl_preds[obs_ids2], norm_axis=1,
            fp=figdir / f'contingency_matrix(acc{test_acc:.1%}).png',
        )
        pl.plot_confus_mat(
            labels_cat[obs_ids1], cl_preds[obs_ids1], classes_on=classes,
            fp=figdir / f'contingency_matrix-train.png',
        )

        # heatmap of predicted probabilities
        gs = pl.wrapper_heatmap_scores(
            df_probs.iloc[obs_ids2], obs.iloc[obs_ids2], ignore_index=True,
            col_label='celltype', col_pred='predicted',
            n_subsample=50,
            cmap_heat='magma_r', #if prob_func == 'softmax' else 'RdBu_r'
            fp=figdir / f'heatmap_probas.pdf'
        )
    return dpair, trainer, h_dict, predictor, ENV_VARs


def gather_came_results(
        dpair,
        trainer: Trainer,
        classes: Sequence,
        keys: Sequence,
        keys_compare=None,
        resdir: Union[str, Path] = '.',
        checkpoint: Union[int, str] = 'best',
        batch_size: Optional[int] = None,
):
    """ Packed function for pipeline as follows:
    1. load the 'best' model weights
    2. get the predictions for cells, including probabilities (from logits)
    3. get and the hidden states for both cells and genes
    3. make a predictor
    """
    resdir = Path(resdir)
    if isinstance(checkpoint, int):
        trainer.load_model_weights(checkpoint)
    elif 'best' in checkpoint.lower():
        trainer.load_model_weights()
    elif 'last' in checkpoint.lower():
        trainer.load_model_weights(trainer._cur_epoch)
    else:
        raise ValueError(
            f'`checkpoint` should be either str ("best" or "last") or int, '
            f'got {checkpoint}'
        )
    out_cell = trainer.get_current_outputs(batch_size=batch_size)['cell']

    out_cell = out_cell.cpu().clone().detach().numpy()
    pd.DataFrame(out_cell[dpair.obs_ids1], columns=classes).to_csv(resdir / "df_logits1.csv")
    pd.DataFrame(out_cell[dpair.obs_ids2], columns=classes).to_csv(resdir / "df_logits2.csv")
    predictor = Predictor(classes=classes).fit(
            out_cell[dpair.obs_ids1],
            detach2numpy(trainer.train_labels),
        )
    predictor.save(resdir / f'predictor.json')

    probas_all = as_probabilities(out_cell, mode='softmax')
    cl_preds = predict_from_logits(probas_all, classes=classes)
    obs = pd.DataFrame(
        {keys[0]: dpair.get_obs_labels(keys, asint=False),
         # true labels with `unknown` for unseen classes in query data
         'celltype': dpair.get_obs_anno(keys_compare),  # labels for comparison
         'predicted': cl_preds,
         'max_probs': np.max(probas_all, axis=1),
         })
    obs['is_right'] = obs['predicted'] == obs[keys[0]]
    df_probs = pd.DataFrame(probas_all, columns=classes)
    dpair.set_common_obs_annos(obs)
    dpair.set_common_obs_annos(df_probs, ignore_index=True)
    dpair.obs.to_csv(resdir / 'obs.csv')
    dpair.save_init(resdir / 'datapair_init.pickle')

    # hidden states are stored in sc.AnnData to facilitated downstream analysis
    h_dict = trainer.model.get_hidden_states()  # trainer.feat_dict, trainer.g)

    # group counts statistics (optinal)
    gcnt = pp.group_value_counts(dpair.obs, 'celltype', group_by='dataset')
    logging.info(str(gcnt))
    gcnt.to_csv(resdir / 'group_counts.csv')
    return obs, df_probs, h_dict, predictor


def preprocess_aligned(
        adatas,
        key_class: str,
        df_varmap_1v1: Optional[pd.DataFrame] = None,
        use_scnets: bool = True,
        n_pcs: int = 30,
        nneigh_scnet: int = 5,
        nneigh_clust: int = 20,
        ntop_deg: int = 50,
        key_clust: str = 'clust_lbs'
):
    """
    Packed function for process adatas with aligned features
    (i.e. one-to-one correspondence).

    Processing Steps:

        * align variables
        * preprocessing
        * candidate genes (HVGs and DEGs)
        * pre-clustering query data
        * single-cell network

    Parameters
    ----------
    adatas:
    key_class
    df_varmap_1v1
    use_scnets
    n_pcs
    nneigh_scnet
    nneigh_clust
    ntop_deg
    key_clust

    Returns
    -------

    """
    adatas = pp.align_adata_vars(
        adatas[0], adatas[1], df_varmap_1v1, unify_names=True,
    )

    logging.info('================ preprocessing ===============')
    params_preproc = dict(
        target_sum=None,
        n_top_genes=2000,
        n_pcs=n_pcs,
        nneigh=nneigh_scnet,
    )
    # NOTE: using the median total-counts as the scale factor
    # (may perform better than fixed number)
    adata1 = pp.quick_preprocess(adatas[0], **params_preproc)
    adata2 = pp.quick_preprocess(adatas[1], **params_preproc)

    # the single-cell network
    if use_scnets:
        scnets = [pp.get_scnet(adata1), pp.get_scnet(adata2)]
    else:
        scnets = None
    # get HVGs
    hvgs1, hvgs2 = pp.get_hvgs(adata1), pp.get_hvgs(adata2)

    # cluster labels
    clust_lbs2 = pp.get_leiden_labels(
        adata2, force_redo=True,
        nneigh=nneigh_clust,
        neighbors_key='clust',
        key_added=key_clust,
        copy=False
    )
    adatas[1].obs[key_clust] = clust_lbs2

    params_deg = dict(n=ntop_deg, force_redo=False,
                      inplace=True, do_normalize=False)
    # adata1&2 have already been normalized before
    degs1 = pp.compute_and_get_DEGs(
        adata1, key_class, **params_deg)
    degs2 = pp.compute_and_get_DEGs(
        adata2, key_clust, **params_deg)
    ###
    vars_feat = list(set(degs1).union(degs2))
    vars_node = list(set(hvgs1).union(hvgs2).union(vars_feat))

    dct = dict(
        adatas=adatas,
        vars_feat=vars_feat,
        vars_as_nodes=vars_node,
        scnets=scnets,
    )
    return dct, (adata1, adata2)


def preprocess_unaligned(
        adatas,
        key_class: str,
        use_scnets: bool = True,
        n_pcs: int = 30,
        nneigh_scnet: int = 5,
        nneigh_clust: int = 20,
        ntop_deg: int = 50,
        key_clust: str = 'clust_lbs',
        node_source: str = 'hvg,deg',
):
    """
    Packed function for process adatas with un-aligned features.
    (i.e. some of them could be one-to-many or many-to-one correspondence)

    Processing Steps:

        * preprocessing
        * candidate genes (HVGs and DEGs)
        * pre-clustering query data
        * single-cell network

    Parameters
    ----------
    adatas
    key_class
    use_scnets
    n_pcs
    nneigh_scnet
    nneigh_clust
    ntop_deg
    key_clust

    Returns
    -------

    """
    logging.info('================ preprocessing ===============')
    params_preproc = dict(
        target_sum=None,
        n_top_genes=2000,
        n_pcs=n_pcs,
        nneigh=nneigh_scnet,
    )
    # NOTE:
    # by default, the original adatas will not be changed
    # using the median total-counts as the scale factor \
    # may perform better than fixed number
    adata1 = pp.quick_preprocess(adatas[0], **params_preproc)
    adata2 = pp.quick_preprocess(adatas[1], **params_preproc)
    # the single-cell network
    if use_scnets:
        scnets = [pp.get_scnet(adata1), pp.get_scnet(adata2)]
    else:
        scnets = None
    # get HVGs
    hvgs1, hvgs2 = pp.get_hvgs(adata1), pp.get_hvgs(adata2)

    # cluster labels
    clust_lbs2 = pp.get_leiden_labels(
        adata2, force_redo=True,
        nneigh=nneigh_clust,
        neighbors_key='clust',
        key_added=key_clust,
        copy=False
    )
    adatas[1].obs[key_clust] = clust_lbs2

    params_deg = dict(n=ntop_deg, force_redo=False,
                      inplace=True, do_normalize=False)
    # adata1&2 have already been normalized before
    degs1 = pp.compute_and_get_DEGs(
        adata1, key_class, **params_deg)
    degs2 = pp.compute_and_get_DEGs(
        adata2, key_clust, **params_deg)

    vars_use = [degs1, degs2]
    node_source = node_source.lower()
    if 'hvg' in node_source and 'deg' in node_source:
        vars_as_nodes = [np.unique(np.hstack([hvgs1, degs1])),
                        np.unique(np.hstack([hvgs2, degs2]))]
    elif 'hvg' in node_source:
        vars_as_nodes = [hvgs1, hvgs2]
    else:
        vars_as_nodes = [degs1, degs2]

    dct = dict(
        adatas=adatas,
        vars_use=vars_use,
        vars_as_nodes=vars_as_nodes,
        scnets=scnets,
    )
    return dct, (adata1, adata2)


def __test1__(n_epochs: int = 5, batch_size=None):
    seed_everything()
    datadir = Path(os.path.abspath(__file__)).parent / 'sample_data'
    sp1, sp2 = ('human', 'mouse')
    dsnames = ('Baron_human', 'Baron_mouse')

    df_varmap_1v1 = pd.read_csv(datadir / f'gene_matches_1v1_{sp1}2{sp2}.csv', )

    dsn1, dsn2 = dsnames
    adata_raw1 = sc.read_h5ad(datadir / f'raw-{dsn1}.h5ad')
    adata_raw2 = sc.read_h5ad(datadir / f'raw-{dsn2}.h5ad')
    adatas = [adata_raw1, adata_raw2]

    key_class = 'cell_ontology_class'
    time_tag = make_nowtime_tag()
    resdir = Path('_temp') / f'{dsnames}-{time_tag}'

    came_inputs, (adata1, adata2) = preprocess_aligned(
        adatas,
        key_class=key_class,
        df_varmap_1v1=df_varmap_1v1,
    )

    _ = main_for_aligned(
        **came_inputs,
        dataset_names=dsnames,
        key_class1=key_class,
        key_class2=key_class,
        do_normalize=True,
        n_epochs=n_epochs,
        resdir=resdir,
        check_umap=not True,  # True for visualizing embeddings each 40 epochs
        n_pass=100,
        params_model=dict(residual=False),
        batch_size=batch_size
    )

    del _
    torch.cuda.empty_cache()
    logging.debug('memory cleared\n')
    print('Test passed for ALIGNED!')


def __test2__(n_epochs: int = 5, batch_size=None):
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

    _ = main_for_unaligned(
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
        params_model=dict(residual=False),
        batch_size=batch_size,
    )

    del _
    torch.cuda.empty_cache()
    logging.debug('memory cleared\n')
    print('Test passed for UN-ALIGNED!')
