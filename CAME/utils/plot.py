# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:31:28 2020

@author: Xingyan Liu
"""
from typing import Union, Mapping, Sequence, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import metrics
import scanpy as sc

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns

from . import _alluvial

"""
Example:
    import utils_plot as uplt
    uplt.heatmap(df)
"""


# In[]

def _save_with_adjust(fig, fpath=None, figsize=None, **kwds):
    if figsize is not None:
        fig.set_size_inches(*figsize)
    if fpath is not None:
        fig.savefig(fpath, bbox_inches='tight', **kwds)
        print(f'figure has been saved into:\n\t{fpath}')
    else:
        fig.show()
    plt.close()


def rotate_xticklabels(ax, angle=45, **kwargs):
    ax.set_xticklabels(ax.get_xticklabels(), rotation=angle,
                       **kwargs)


def rotate_yticklabels(ax, angle=45, **kwargs):
    ax.set_yticklabels(ax.get_yticklabels(), rotation=angle,
                       **kwargs)


# In[]
# colors
def view_color_map(cmap='viridis', n=None, figsize=(6, 2), s=150, k=20,
                   ax=None,
                   grid=False, **kwds):
    """
    n: total number of colors
    k: number of colors to be plotted on each line.
    
    Examples
    --------
    import funx as fx
    colors = ['Set1', 'viridis', 'Spectral']
    fx.view_color_map(colors[-1], n=20)
    
    cmap = sc.pl.palettes.zeileis_26
    cmap = sc.pl.palettes.default_64
    fx.view_color_map(cmap, k=16)    
    """
    if not isinstance(cmap, (np.ndarray, list)):
        #        from matplotlib import cm
        cmp = mpl.cm.get_cmap(cmap, n)
        n = 20 if n is None else n
        cmp = [cmp(i) for i in range(n)]
    else:
        n = len(cmap) if n is None else n
        cmp = cmap
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for i in range(n):
        ax.scatter(i % k, i // k, color=cmp[i], s=s)
    plt.grid(b=grid, )  # axis='y')
    plt.show()
    return ax


def get_colors(cmap='Spectral', n=5, to_hex=True):
    """
    fx.get_colors('Reds', 4)
    fx.view_color_map('Reds', 4)
    """
    #    import matplotlib.colors as mcolors
    cmp = plt.cm.get_cmap(cmap, n)
    colors = [cmp(i) for i in range(n)]
    if to_hex:
        return [mcolors.to_hex(c) for c in colors]
    else:
        return colors


def diy_cmap_grey_bg(name_fg='RdPu', low=0.15, rm_high=0.01, n=100):
    s = int(n * low)
    t = max((1, int(n * rm_high)))
    print((s, t))
    candi_colors = ['#d9d9d9'] * s + get_colors(name_fg, n)[s: -t]
    cmap = mcolors.ListedColormap(candi_colors)
    return cmap


# In[]

"""     Visualization of group correspondances on an alluvial plot
=======================================================================
"""


def _alluvial_dict_from_confusdf(df: pd.DataFrame, vals_ignore=[0, np.nan]):
    dct = {}
    rowsum = df.sum(1)

    for i, idx in enumerate(df.index):
        #        print(i, idx, type(rowsum[idx]), rowsum[idx])
        if rowsum[idx] in vals_ignore:
            continue
        dct[idx] = {}
        for col in df.columns:
            val = df.loc[idx, col]
            if val > 0:
                dct[idx][col + ' '] = val  # make sure no collisions for names in each side
    return dct


def alluvial_plot(confsdf: pd.DataFrame,
                  labels=['', ''],  # ['true', 'predicted'],
                  label_shift=0,  # -18
                  title='alluvial plot',
                  alpha=0.75,
                  cmap_name='tab20_r',
                  shuffle_colors=True,
                  fonsize_title=11,
                  figsize=(5, 5),
                  fname=None,
                  **kwds):
    """ visualizing confusion matrix using alluvial plot
    """
    input_dct = _alluvial_dict_from_confusdf(confsdf)
    cmap = mpl.cm.get_cmap(cmap_name)
    ax = _alluvial.plot(input_dct, labels=labels,
                        label_shift=label_shift,
                        cmap=cmap, alpha=alpha,
                        shuffle_colors=shuffle_colors,
                        figsize=figsize,
                        **kwds)
    ax.set_title(title, fontsize=fonsize_title)
    fig = ax.get_figure()
    #    fig.set_size_inches(*figsize)
    plt.show()
    _save_with_adjust(fig, fname, )
    return ax


# In[]
"""     functions for plotting confusion or contingency matrix
==================================================================
"""


def plot_contingency_mat(
        y_true, y_pred,
        norm_axis=1,
        arrange: bool = True,
        order_rows: bool = True,
        order_cols: bool = False,
        ax=None,
        figsize: tuple = (4, 3),
        fp: Union[Path, str, None] = None,
        **kwds
):
    from .analyze import wrapper_contingency_mat
    contmat = wrapper_contingency_mat(
        y_true, y_pred, normalize_axis=norm_axis,
        order_rows=order_rows, order_cols=order_cols, )
    if arrange:
        # align column- and row- names as possible
        from .analyze import arrange_contingency_mat
        contmat = arrange_contingency_mat(contmat)

    ax = heatmap(contmat, figsize=figsize, ax=ax, fp=fp, **kwds)

    return ax, contmat


def plot_confus_mat(y_true, y_pred, classes_on=None,
                    normalize='true',
                    linewidths=0.02, linecolor='grey',
                    figsize: tuple = (4, 3),
                    ax=None, fp=None,
                    **kwargs):
    """ by default, normalized by row (true classes)
    """
    if classes_on is None:
        classes_on = list(set(y_true).union(y_pred))

    mat = metrics.confusion_matrix(y_true, y_pred, labels=classes_on,
                                   normalize=normalize)
    # return sns.heatmap(mat, linewidths=linewidths, linecolor=linecolor,
    #                    xticklabels=classes_on, yticklabels=classes_on,
    #                    **kwargs)
    mat = pd.DataFrame(data=mat, index=classes_on, columns=classes_on)
    ax = heatmap(mat, figsize=figsize, ax=ax, fp=fp,
                 linewidths=linewidths, linecolor=linecolor,
                 **kwargs)
    return ax, mat


def plot_confus_multi_mats(ytrue_lists, ypred_lists, classes_on=None,
                           nrows=1, ncols=2, figsize=(8, 3),
                           vmax=1,
                           fname=None):
    """
    combime multiple confusion matrix-plots into a single plot
    """
    fig, axs = plt.subplots(nrows, ncols, sharey=True, figsize=figsize)
    for k in range(len(ytrue_lists)):
        if nrows == 1 or ncols == 1:
            ax = axs[k]
        else:
            irow = k // ncols
            icol = k % nrows
            ax = axs[irow, icol]
        plot_confus_mat(ytrue_lists[k], ypred_lists[k],
                        classes_on=classes_on,
                        square=True,  # cbar=False,
                        vmax=vmax,
                        ax=ax,
                        )
    _save_with_adjust(fig, fname)

    return axs


# In[]
""" plot functions (for inspecting training process)
"""


def plot_line_list(ys, lbs=None,
                   ax=None, figsize=(4.5, 3.5),
                   tt=None, fp=None,
                   legend_loc=(1.05, 0),
                   **kwds):
    """
    ys: a list of lists, each sub-list is a set of curve-points to be plotted.
    """
    if lbs is None:
        lbs = list(map(str, range(len(ys))))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for i, y in enumerate(ys):
        ax.plot(y, label=lbs[i], **kwds)
    ax.legend(loc=legend_loc)
    if tt is not None:
        ax.set_title(tt)
    _save_with_adjust(ax.figure, fp, )

    return ax


def plot_records_for_trainer(
        trainer, record_names, start=0, end=None,
        lbs=None, tt='training logs', fp=None,
        **kwds):
    if lbs is None:
        lbs = record_names
    if end is not None:
        end = int(min([trainer._cur_epoch + 1, end]))
    line_list = [getattr(trainer, nm)[start: end] for nm in record_names]

    return plot_line_list(line_list, lbs=lbs, tt=tt, fp=fp, **kwds)


def venn_plot(sets, set_labels=None, regular=False,
              tt='Venn plot', ax=None, fp=None, **kwds):
    """
    sets: iterable
    set_labels: list[str]
    regular: bool, only for sets of strings!
        wether to regularize the strings in sets 
    """
    from matplotlib_venn import venn2, venn3
    if regular:
        print('Regularizing strings in sets (UPPER case ignored)')
        lowerStr = lambda x: [str(c).lower() for c in list(x)]
        sets = list(map(set, map(lowerStr, sets)))
    if not isinstance(sets[0], set):
        # assure that each item in `sets` is a set object
        sets = list(map(set, sets))
    n = len(sets)
    if set_labels is None:
        set_labels = np.arange(n)
    venn = {2: venn2, 3: venn3}[n]
    if ax is None:
        fig, ax = plt.subplots()
    venn(sets, set_labels=set_labels, ax=ax, **kwds)
    ax.set_title(tt)

    _save_with_adjust(ax.figure, fp, )
    #    plt.show()

    return ax


# In[]


def heatmap(df_hmap: pd.DataFrame,
            norm_method: Union[None, str] = None,
            norm_axis=1,
            order_row=False,
            order_col=False,
            figsize=(6, 6),  # ignored if ax is not None
            ax=None,
            xlabel=None,
            ylabel=None,
            cmap='magma_r',  # 'RdBu_r'
            cbar=True,
            cbar_kws={'shrink': 0.5},
            fp=None,
            xrotation=45,
            yrotation=None,
            **kwds):
    """ wrapper of sns.heatmap()
    ===================================
    norm_method: None, 'zs', 'max', 'maxmin'
    norm_axis: 0 or 1. 0 for column-normalization and 1 for rows.
    figsize: tuple, (6, 6) by default, ignored if ax is not None.
    
    
    """
    # normalize values
    if norm_method == 'max':  # ALL non-negative values are required!
        if (df_hmap < 0).any():
            raise ValueError('Data with ALL non-negative values are required '
                             'for "max-normalization"')
        df_hmap = df_hmap.apply(lambda x: x / x.max() if x.max() > 0 else x,
                                axis=norm_axis)
    elif norm_method == 'maxmin':
        eps = 1e-8
        df_hmap = df_hmap.apply(
            lambda x: (x - x.min()) / (x.max() - x.min() + eps),
            axis=norm_axis)
    elif norm_method == 'zs':
        from .preprocess import zscore
        if norm_axis == 0:
            df_hmap = zscore(df_hmap)  # column normalization
        elif norm_axis == 1:
            df_hmap = zscore(df_hmap.T).T

    # re-order rows or columns
    from .preprocess import order_contingency_mat
    if order_col:
        df_hmap = order_contingency_mat(df_hmap, axis=0)
    if order_row:
        df_hmap = order_contingency_mat(df_hmap, axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_hmap,
                # yticklabels=False,
                cbar=cbar,
                cbar_kws=cbar_kws,
                ax=ax,
                cmap=cmap,
                # col_colors = cl_color_match,
                **kwds)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xrotation is not None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xrotation,
                           ha='right' if xrotation < 90 else 'center'
                           )
    if yrotation is not None:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=yrotation)

    if fp is not None:
        _save_with_adjust(ax.figure, fp)
    return ax


def heatmap_probas(
        df_data, lbs,  # sort_lbs=False,
        name_label=None,
        cmap_heat='magma_r', cmap_lb='tab20',
        figsize=(8, 4),
        vmax=1, vmin=0,
        xrotation=30,
        fp=None):
    """ heatmap showing probabilities
    """
    lbs = pd.Categorical(lbs, )
    pix_lbs = np.vstack([lbs.codes] * 2)

    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = fig.add_gridspec(
        nrows=12, ncols=80, left=0.05, right=0.98,
        hspace=0.25, wspace=2)
    w_cbar = 2
    ax1 = fig.add_subplot(gs[:-1, : -w_cbar])
    ax11 = fig.add_subplot(gs[:-1, -w_cbar:])
    ax2 = fig.add_subplot(gs[-1:, : -w_cbar])
    im = ax1.imshow(df_data,
                    interpolation='nearest',
                    cmap=cmap_heat,
                    # origin='lower', #extent=[-3, 3, -3, 3],
                    aspect='auto',
                    vmax=1, vmin=0)
    fig.colorbar(mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap=cmap_heat,
    ),
        cax=ax11, )
    # im = ax1.pcolor(df_data, cmap='magma_r', vmax=1, vmin=0)
    ax2.pcolor(pix_lbs, cmap=plt.get_cmap(cmap_lb))

    for _ax in [ax1, ax2, ]:
        _ax.grid(False)
    ax1.set_yticks(np.arange(df_data.shape[0]))
    ax1.set_yticklabels(df_data.index)
    ax1.set_xticks([])
    ax2.set_yticks([1])
    ax2.set_yticklabels([name_label])

    # setting class-labels (x-ticks)
    classes = pd.unique(lbs)
    cut_loc = np.hstack([[0], np.flatnonzero(np.diff(lbs.codes)), [len(lbs)]])  # // 2
    x_loc = cut_loc[:-1] + np.diff(cut_loc) // 2
    ax2.set_xticks(x_loc)
    if xrotation:
        ax2.set_xticklabels(classes, rotation=xrotation, ha='right')
    else:
        ax2.set_xticklabels(classes, )

    for loc in ['right', 'top', 'bottom', 'left'][:-1]:
        ax1.spines[loc].set_visible(False)
        ax11.spines[loc].set_visible(False)
        ax2.spines[loc].set_visible(False)
    #    _ax.set_axis_off()
    if fp:
        _save_with_adjust(fig, fp)
    return gs


def grid_display_probas(
        df,
        labels,
        classes=None,
        figsize=(6, 5),
        sharey=True,
):
    """ violin plots of the distributions """
    if classes is None:
        classes = list(set(labels))
    fig, axs = plt.subplots(
        len(classes), 2, figsize=figsize,
        sharex=True, sharey=sharey,
        gridspec_kw={'hspace': 0.0, 'wspace': 0.}
    )
    for i, cl in enumerate(classes):
        df_sub = df[labels == cl]
        y_mid = 0.4  # (df_sub.max() + df_sub.min()) / 2
        sns.violinplot(data=df_sub, ax=axs[i, 1], linewidth=.01, vmin=0)
        axs[i, 1].set_ylim(-0.1, 1.1)
        axs[i, 0].text(df.shape[1] - 1, y_mid, cl, ha='right')
        axs[i, 0].set_axis_off()
    rotate_xticklabels(axs[-1, 1], ha='right')
    return fig


def wrapper_heatmap_scores(
        df_score: pd.DataFrame,
        obs: pd.DataFrame,
        col_label: str = 'celltype',
        col_pred: str = 'predicted',
        figsize: tuple = (5, 3),
        n_subsample: Optional[int] = 50,
        ignore_index: bool = False,
        fp=None,
        **kwds
):
    """ sort columns and rows, plot heatmap of celltype scores """
    cols_anno = [col_label, col_pred]
    df_lbs = obs[cols_anno]
    if ignore_index:
        df_lbs = df_lbs.copy()
        df_lbs.index = df_score.index
    df_lbs = df_lbs.sort_values(cols_anno)
    if n_subsample:
        from .base import subsample_each_group
        indices = subsample_each_group(df_lbs[col_label], n_out=n_subsample)
    else:
        indices = df_lbs.index
    cols_ordered = [c for c in sorted(df_lbs[col_pred].unique())
                    if c in df_score.columns]
    
    df_data = df_score.loc[indices][cols_ordered].copy()
    lbs = df_lbs[col_label].loc[indices]

    gs = heatmap_probas(
        df_data.T, lbs, name_label='true label',
        figsize=figsize, fp=fp, **kwds
    )
    return gs


# In[]
def _get_affine_mat(angle_x=30, angle_y=150):
    """ rotate x and y axis to mock 3D projection
    if angle_x=0, angle_y=90, an identity matrix will be returned.
    """
    angle_x, angle_y = tuple(map(lambda x: (x / 180) * np.pi, (angle_x, angle_y)))
    _transmat = np.array([[np.cos(angle_x), np.sin(angle_x)],
                          [np.cos(angle_y), np.sin(angle_y)]
                          ])
    return _transmat


def _plot_edges(sids, tids,
                xy1, xy2=None,
                ax=None,
                figsize=(4, 4), **kwds_edge):
    """
    sids: sequence of source ids, of shape (n_edges, )
    tids: sequence of target ids, of shape (n_edges, )
    xy1: x, y coordinates of source node (ids), of shape (n1, 2) or (n1+n2, 2)
    xy2: x, y coordinates of target node (ids)
    """
    xy2 = xy1 if xy2 is None else xy2
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)  # w*h
    _kwds_edge = dict(c='gray', alpha=0.5, linewidth=0.08)
    _kwds_edge.update(kwds_edge)
    for pt1, pt2 in zip(xy1[sids], xy2[tids]):
        ax.plot(*zip(pt1, pt2), **_kwds_edge)
    return ax


def plot_edges_by_adj(adj, pos, ax=None, **kwds_edge):
    """ adj: a sparse matrix, None is allowed"""
    adj = sparse.triu(adj).tocoo()
    sids, tids = adj.row, adj.col
    return _plot_edges(sids, tids, xy1=pos, xy2=None, ax=ax, **kwds_edge)


def plot_mapped_graph(
        xy1, xy2, mapping, adj1=None, adj2=None,
        pt_color1=None, pt_color2=None,
        cmap_pt='tab20',
        angle_x=30, angle_y=150,
        offset_scale=1.2,
        yscale=1.,
        pt_size=10,
        style_name='default',
        figsize=(5, 7),
        fp=None,
        **kwds_edge):
    """
    pt_color1, pt_color2: RGB colors or numerical label sequence
    """
    #    style_name = ['dark_background', 'default'][1]
    plt.style.use(style_name)
    _kwds_edge = dict(c='white' if style_name == 'dark_background' else 'grey',
                      alpha=0.35, linewidth=0.08)
    _kwds_edge.update(kwds_edge)

    _transmat = _get_affine_mat(angle_x, angle_y)
    xy1 = xy1.dot(_transmat)
    xy2 = xy2.dot(_transmat)

    x1, y1 = xy1[:, 0], xy1[:, 1]
    x2, y2 = xy2[:, 0], xy2[:, 1]
    y1 *= yscale
    y2 *= yscale
    y1 += max(y2) * offset_scale

    fig, ax = plt.subplots(figsize=figsize)  # w*h
    if adj1 is not None:
        plot_edges_by_adj(adj1, xy1, ax=ax, zorder=0.9, **_kwds_edge)
    if adj2 is not None:
        plot_edges_by_adj(adj2, xy2, ax=ax, zorder=0.9, **_kwds_edge)
    plot_edges_by_adj(mapping, np.vstack([xy1, xy2]), ax=ax, zorder=1, **_kwds_edge)

    cmap_pt = cmap_pt if isinstance(cmap_pt, (list, tuple)) else [cmap_pt] * 2
    ax.scatter(x1, y1, marker='.', c=pt_color1, s=pt_size, cmap=cmap_pt[0], zorder=2.5)
    ax.scatter(x2, y2, marker='.', c=pt_color2, s=pt_size, cmap=cmap_pt[1])
    ax.set_axis_off()

    _save_with_adjust(fig, fp, dpi=400)
    plt.style.use('default')
    return ax


# In[]
def umap_grid(adatas, colors=None,
              ncols=1, figsize=None,
              sharex=True, sharey=True,
              fp=None, **kwds):
    """ multiple-plots of embeddings
    """
    n = len(adatas)
    if isinstance(colors, str) or colors is None:
        colors = [colors] * n

    nrows = n // ncols + min([n % ncols, 1])
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize,
                            sharex=sharex, sharey=sharey)
    for ax, adt, color in zip(axs.flatten(), adatas, colors):
        sc.pl.umap(adt, color=color, ax=ax, show=False,
                   **kwds)

    _save_with_adjust(fig, fp)
    return fig


def plot_splitted_umaps(adata, splitby,
                        left_groups=None,
                        colors=None, ncols=1,
                        figsize=(4, 8), **kwds):
    """ multiple-plots of embeddings
    """
    from .preprocess import bisplit_adata
    adt1, adt2 = bisplit_adata(adata, splitby, left_groups)

    return umap_grid(
        [adt1, adt2], colors=colors,
        ncols=ncols, figsize=figsize, **kwds)


def triple_umaps(adata1, adata2,
                 titles=None,
                 colors=['dataset', 'is_private', 'is_private'],
                 figsize=(6, 4.5),
                 fp=None, **kwds):
    """
    Specific function for gene embeddings only, to be generized
    """
    if titles is None:
        titles = [titles] * 3

    # point sizes
    n1, n2 = adata1.shape[0], adata2.shape[1]
    size0 = int(100000 / (n1 + n2))
    size = size0 / 2

    # color maps
    color_bg = 'lightgray'  # background
    color_fg1 = 'tab:blue'
    color_fg2 = 'lightcoral'  # 'tab:pink'

    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        left=0.05, right=0.95,
        hspace=0.15, wspace=0.1)
    ax1 = fig.add_subplot(gs[:, : -1])
    ax2 = fig.add_subplot(gs[: 1, - 1:])
    ax3 = fig.add_subplot(gs[1:, - 1:])

    sc.pl.umap(adata1, color=colors[0],
               size=size0,
               #           title=f'merged {sp1} and {sp2} genes',
               palette=[color_fg1],
               ax=ax1, show=False, **kwds)
    sc.pl.umap(adata2, color=colors[0], size=size0,
               palette=[color_fg2],
               title=titles[0],
               ax=ax1, show=False, **kwds)
    sc.pl.umap(adata1, color=colors[1], palette=[color_bg, color_fg1],
               title=titles[1],
               size=size, ax=ax2, show=False, **kwds, )
    sc.pl.umap(adata2, color=colors[2], palette=[color_bg, color_fg2],
               title=titles[2],
               size=size, ax=ax3, show=False, **kwds)
    # ax1.legend_.remove()#ax1.legend([])
    # fig
    for _ax in [ax1, ax2, ax3]:
        _ax.grid(False)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.set_axis_off()
        _ax.legend_.remove()

    _save_with_adjust(fig, fp)
    return fig


# In[]
""" annotate texts for genes of interest
"""


def _get_value_index(srs: pd.Series, vals: Sequence, ):
    isin = srs.isin(vals)
    return isin[isin].index


def _get_value_index_list(lst, vals, return_vals=False):
    """
    getting the indexes only the first-occurred one in the list
    """
    lst = list(lst)
    ids = []
    _vals = []
    for v in vals:
        try:
            i = lst.index(v)
            ids.append(i)
            _vals.append(v)
        except ValueError:
            continue
    if return_vals:
        return ids, _vals
    return ids


def umap_with_annotates(
        adt: sc.AnnData,
        text_ids: Sequence,
        color: Optional[str] = None,
        anno_df: Optional[pd.DataFrame] = None,
        text_col: Optional[str] = None,
        index_col: Optional[str] = None,
        anno_fontsize: int = 12,
        ax=None,
        fp=None,
        **plkwds):
    from adjustText import adjust_text
    anno_df = adt.obs if anno_df is None else anno_df

    index_all = anno_df.index if index_col is None else anno_df[index_col]
    idnums, _text_ids = _get_value_index_list(
        index_all, text_ids, return_vals=True)  # only the first one
    # subsetting xumap and df
    xumap = adt.obsm['X_umap'][idnums, :]
    #    anno_df = anno_df.iloc[idnums, :]
    xs = xumap[:, 0]
    ys = xumap[:, 1]
    texts = _text_ids if text_col is None else anno_df[text_col][idnums]
    #    print(texts)

    if ax is None:
        fig, ax = plt.subplots()
    sc.pl.umap(adt, color=color, ax=ax, show=False, **plkwds)
    # annotation
    _texts = [ax.text(x, y, t, {'fontsize': anno_fontsize}) for x, y, t in zip(xs, ys, texts)]
    adjust_text(_texts, ax=ax, arrowprops=dict(arrowstyle='-', color='k'))
    _save_with_adjust(fig, fp)
    return ax


# In[]
def plot_distance_lines(
        dists, names, xlabel='distance',
        markersize=10, fp=None,
):
    """
    names: a list of str, yticklabels
    """

    n = len(names)
    ys1 = [x + 0.9 for x in range(n)]
    ys2 = ys1
    xs1 = [0] * n
    xs2 = dists

    fig, ax = plt.subplots(figsize=(4.5, 0.4 * n + 0.4))
    for i in range(n):
        _x = [xs1[i], xs2[i]]
        _y = [ys1[i], ys2[i]]
        plt.plot(_x, _y, marker='o', markersize=markersize)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.grid(False)
    ax.set_yticks(ys1)
    ax.set_yticklabels(list(names), fontsize=14)
    ax.set_ylim(0.0, n + 0.9)
    # ax.get_yticklabels()
    for loc in ['right', 'top']:
        ax.spines[loc].set_visible(False)

    _save_with_adjust(fig, fp)
    return ax


# In[]
""" Visualization of the abstracted graph
"""


def _adjust_xlims(ax, scale=1.):
    if isinstance(scale, Sequence):
        xlims = ax.get_xlim()
        ax.set_xlim((xlims[0] * scale[0], xlims[1] * scale[1]))
    else:
        xlims = ax.get_xlim()
        ax.set_xlim((xlims[0] * scale, xlims[1] * scale))


def _adjust_ylims(ax, scale=1.):
    if isinstance(scale, Sequence):
        ylims = ax.get_ylim()
        ax.set_ylim((ylims[0] * scale[0], ylims[1] * scale[1]))
    else:
        ylims = ax.get_ylim()
        ax.set_ylim((ylims[0] * scale, ylims[1] * scale))


def plot_multipartite_graph(
        g,
        node_list=None,
        subset_key='subset',
        weight_key='weight',
        nodelb_key='original name',
        nodesz_key='size',
        node_color='lightblue',
        alpha=0.85,
        figsize=(10, 8), fp=None,
        xscale=1.5, yscale=1.1,
        edge_scale=5.,
        node_scale=1.,
        node_size_min=100,
        node_size_max=1800,
        with_labels=True,
        **kwds):
    import networkx as nx
    if node_list is None: node_list = list(g)

    pos = multipartite_layout(g, subset_key=subset_key, )

    edge_vals = [v * edge_scale for v in nx.get_edge_attributes(g, weight_key).values()]
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_axis_off()
    ax.set_frame_on(False)

    nx.draw_networkx_edges(
        g, pos,
        width=edge_vals,
        #            edge_color='grey',
        edge_color=edge_vals, edge_cmap=_cut_cmap('Greys', ),
        alpha=alpha, )
    nodedf = _prepare_for_nxplot(g, **kwds)

    #    node_sizes = np.array([nx.get_node_attributes(g, nodesz_key)[nd] for nd in node_list])
    node_sizes = np.maximum(nodedf['plt_size'] * node_scale, node_size_min)
    #    node_sizes = np.minimum(node_sizes, node_size_max)
    node_color = nodedf['plt_color'].tolist()

    nx.draw_networkx_nodes(g, pos, node_color=node_color, node_size=node_sizes)
    if with_labels:
        node_labels = nx.get_node_attributes(g, nodelb_key)
        tmp = nx.draw_networkx_labels(g, pos, node_labels)
    #    ax = plot_graph(g, pos)
    if xscale is not None:
        _adjust_xlims(ax, xscale)
    #        xlims = ax.get_xlim()
    #        ax.set_xlim((xlims[0] * xscale, xlims[1] * xscale))
    if yscale is not None:
        _adjust_ylims(ax, yscale)

    _save_with_adjust(fig, fp)
    return ax


def _prepare_for_nxplot(
        g, names=['cell group', 'gene module'],
        sizes=[1800, 900, 900, 1800],
        colors=['pink', 'lightblue', 'lightblue', 'pink']):
    """
    g: a multipartite graph with 4 layers
    """
    nodedf = pd.DataFrame(g.nodes.values(), index=g.nodes.keys())
    nodedf['ntype'] = nodedf['subset'].apply(
        lambda x: names[0] if x in [0, 3] else names[1]
    )
    nodedf['plt_color'] = nodedf['subset'].apply(
        lambda x: colors[x]
    )
    nodedf['plt_size'] = nodedf.groupby('ntype')['size'].apply(lambda x: x / x.max())

    for isubset, size_scale in enumerate(sizes):
        inds_sub = nodedf['subset'] == isubset
        nodedf['plt_size'][inds_sub] *= size_scale
    return nodedf


def _cut_cmap(cmap='Greys', low=0.2, high=0.9):
    cmap0 = plt.cm.get_cmap(cmap)
    colors = cmap0(np.linspace(0, 1, 256))
    colors = colors[int(256 * low): int(256 * high)]
    cmap_new = mcolors.ListedColormap(colors)
    return cmap_new


# In[]
"""     graph layout 
(copied from `networkx.drawing.layout.py`, with some modifications)
"""


def _process_params(G, center, dim):
    # Some boilerplate code.
    import networkx as nx
    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center


def multipartite_layout(
        G, subset_key="subset",
        align="vertical",
        spread_nodes=True,  # xingyan
        scale=1, center=None):
    """Position nodes in layers of straight lines.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    subset_key : string (default='subset')
        Key of node data to be used as layer subset.

    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.complete_multipartite_graph(28, 16, 10)
    >>> pos = nx.multipartite_layout(G)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    Network does not need to be a complete multipartite graph. As long as nodes
    have subset_key data, they will be placed in the corresponding layers.

    """
    import numpy as np

    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}

    layers = {}
    for v, data in G.nodes(data=True):
        try:
            layer = data[subset_key]
        except KeyError:
            msg = "all nodes must have subset_key (default='subset') as data"
            raise ValueError(msg)
        layers[layer] = [v] + layers.get(layer, [])

    nodes_per_layer = [len(item[1]) for item in layers.items()]
    print(nodes_per_layer)
    max_nodes = max(nodes_per_layer)
    thd = max_nodes * int(spread_nodes)
    pos = None
    nodes = []
    if align == "vertical":
        width = len(layers)
        for i, layer in layers.items():
            n_nodes = len(layer)
            height = max([n_nodes, thd])
            xs = np.repeat(i, n_nodes)
            ys = np.linspace(0, height, n_nodes, dtype=float)
            offset = ((width - 1) / 2, (height - 1) / 2)
            layer_pos = np.column_stack([xs, ys]) - offset
            if pos is None:
                pos = layer_pos
            else:
                pos = np.concatenate([pos, layer_pos])
            nodes.extend(layer)
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(nodes, pos))
        return pos

    if align == "horizontal":
        height = len(layers)
        for i, layer in layers.items():
            n_nodes = len(layer)
            width = max([n_nodes, thd])
            xs = np.linspace(0, width, n_nodes, dtype=float)
            ys = np.repeat(i, width)
            offset = ((width - 1) / 2, (height - 1) / 2)
            layer_pos = np.column_stack([xs, ys]) - offset
            if pos is None:
                pos = layer_pos
            else:
                pos = np.concatenate([pos, layer_pos])
            nodes.extend(layer)
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(nodes, pos))
        return pos

    msg = "align must be either vertical or horizontal."
    raise ValueError(msg)


def rescale_layout(pos, scale=1):
    """Returns scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.

    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.

    See Also
    --------
    rescale_layout_dict
    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos
