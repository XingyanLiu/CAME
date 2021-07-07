# -*- coding: UTF-8 -*-
"""
@CreateDate: 2021/06/23
@Author: Xingyan Liu
@File: plot_uk.py
@Project: CAME
"""
import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import scanpy as sc

import CAME
from CAME import pipeline, pp, pl


def plot_uk_results(
        resdir,
        p: float = 5e-4,
        prob_func: str = 'sigmoid',
        trans_mode=2,
        fig_types=('pdf', 'svg'),
):
    """

    Parameters
    ----------
    resdir
    p: p-value cut-off
        lower for stricter rejection,
    prob_func:
        {'sigmoid', 'softmax'}
    fig_types: str or a tuple of strings
    """
    resdir = Path(resdir)
    figdir = resdir / 'figs'
    if isinstance(fig_types, str):
        fig_types = [fig_types]
        
    # load logits, metadata and predictor
    df_logits2 = pd.read_csv(resdir / 'df_logits2.csv', index_col=0)
    
    predictor = CAME.Predictor.load(resdir / 'predictor.json')

    predictor.save(resdir / 'predictor-0.json') #backup

    obs = pd.read_csv(resdir / 'obs.csv', index_col=0)

    ########################################
    dpair, model = CAME.load_dpair_and_model(resdir)
    labels, classes = dpair.get_obs_labels(
        "cell_ontology_class", add_unknown_force=False)
    classes = df_logits2.columns
    predictor = CAME.Predictor(classes=classes)
    predictor.fit(
        pd.read_csv(resdir / 'df_logits1.csv', index_col=0).values,
        labels[dpair.obs_ids1],
    )
    predictor.save(resdir / 'predictor.json')
    # ########################################

    dsn2 = obs['dataset'].iloc[-1]
    obs2 = obs[obs['dataset'] == dsn2]

    pred_test = predictor.predict(
        df_logits2.values, p=p, trans_mode=trans_mode)
    print(pd.value_counts(pred_test))
    y_true = obs2['celltype'].values

    # compute, re-order, and plot contingency matrix
    ax, contmat = pl.plot_contingency_mat(y_true, pred_test, norm_axis=1)
    for ftype in fig_types:
        pl._save_with_adjust(
            ax.figure, figdir / f"contmat-{trans_mode}-{p:.1e}.{ftype}")
    logging.warning(contmat)
    
    classes2 = [c for c in contmat.columns if c in df_logits2.columns]
    logging.warning(f"classes2={classes2}")
    df_probas = pd.DataFrame(
        data=CAME.as_probabilities(df_logits2, mode=prob_func),
        # data=predictor.predict_pvalues(df_logits2.values),
        columns=df_logits2.columns
    )

    fig = pl.grid_display_probas(
        df_probas, y_true, contmat.index, 
        figsize=(6, 6))
    for ftype in fig_types:
        pl._save_with_adjust(fig, figdir / f"vlnGrid-{prob_func}.{ftype}")

    gs = pl.wrapper_heatmap_scores(
        df_probas, obs2, ignore_index=True,
        cmap_heat='magma_r' if prob_func == 'softmax' else 'RdBu_r'
    )
    for ftype in fig_types:
        pl._save_with_adjust(gs.figure, figdir / f"heatMap-{prob_func}.{ftype}")


def plot_all(dirname,
             p,
             prob_func,
             trans_mode=3,
             fig_types=('pdf', 'svg')
             ):
    subdirs = os.listdir(dirname)
    for subdir in subdirs:
        type_rm = subdir.split("-")[-1]
        resdir = dirname / subdir
        print(type_rm.center(60, '='))
        print(resdir)
        print(f'p={p:.1e}, {prob_func}\n')
        plot_uk_results(
            resdir, p=p, prob_func=prob_func, 
            trans_mode=trans_mode,
            fig_types=fig_types)


# In[]

dirname = Path("../_temp/('Baron_human', 'Baron_mouse')-(06-20 19.49.07)")
dirname0 = Path("_case_res")
dirname = dirname0 / "uk-('Lake_2018', 'Tasic18')(06-23 11.45.53)"
dirname = dirname0 / "uk-('Lake_2018', 'Tasic18')(06-23 14.37.55)"
dirname = dirname0 / "uk-('Lake_2018', 'Tasic18')(06-24 09.57.28)"
# dirname = dirname0 / "uk-('Lake_2018', 'Tosches_turtle')(06-24 10.31.44)"
dirname = dirname0 / "uk-('Lake_2018', 'Tosches_turtle')(06-28 01.13.23)"
subdirs = os.listdir(dirname)
plot_all(dirname, 5e-3, 'sigmoid', trans_mode=3)



