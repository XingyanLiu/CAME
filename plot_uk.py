# -*- coding: UTF-8 -*-
"""
@CreateDate: 2021/06/23
@Author: Xingyan Liu
@File: plot_uk.py
@Project: CAME
"""
import os
from pathlib import Path

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
        fig_types=('pdf', 'svg')
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
    obs = pd.read_csv(resdir / 'obs.csv', index_col=0)

    dsn2 = obs['dataset'].iloc[-1]
    obs2 = obs[obs['dataset'] == dsn2]

    pred_test = predictor.predict(
        df_logits2.values, p=p, trans_mode=3)
    print(pd.value_counts(pred_test))
    y_true = obs2['celltype'].values

    # compute, re-order, and plot contingency matrix
    ax, contmat = pl.plot_contingency_mat(y_true, pred_test, norm_axis=1)
    for ftype in fig_types:
        pl._save_with_adjust(ax.figure, figdir / f"contmat-{p:.1e}.{ftype}")

    df_probas = pd.DataFrame(
        data=CAME.as_probabilities(df_logits2, mode=prob_func),
        # data=predictor.predict_pvalues(df_logits2.values),
        columns=df_logits2.columns
    )[contmat.columns[:-1]]

    fig = pl.grid_display_probas(df_probas, y_true, contmat.index)
    for ftype in fig_types:
        pl._save_with_adjust(fig, figdir / f"vlnGrid-{prob_func}.{ftype}")

    fig = pl.wrapper_heatmap_scores(
        df_probas, obs2, ignore_index=True
    )
    for ftype in fig_types:
        pl._save_with_adjust(fig, figdir / f"heatMap-{prob_func}.{ftype}")


def plot_all(dirname,
             p,
             prob_func,
             fig_types=('pdf', 'svg')):
    subdirs = os.listdir(dirname)
    for subdir in subdirs:
        type_rm = subdir.split("_")[-1]
        resdir = dirname / subdir
        print(type_rm.center(60, '='))
        print(resdir)
        print(f'p={p:.1e}, {prob_func}\n')
        plot_uk_results(resdir, p=p, prob_func=prob_func, fig_types=fig_types)


# In[]

dirname = Path("../_temp/('Baron_human', 'Baron_mouse')-(06-20 19.49.07)")
subdirs = os.listdir(dirname)
plot_all(dirname, 5e-4, 'softmax')
