# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:51:20 2021

@author: Xingyan Liu
"""

from typing import Union, Mapping, Sequence
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


def _save_with_adjust(fig, fpath=None, figsize=None, **kwds):
    
    if figsize is not None:
        fig.set_size_inches(*figsize)
    if fpath is not None:
        fig.savefig(fpath, bbox_inches = 'tight', **kwds)
        print(f'figure has been saved into:\n\t{fpath}')
    else:
        fig.show()
    plt.close()


def plot_pure_umap(
        adata, color, 
        ax=None, figsize=(4, 4), 
        tag='', title='',
        figdir=Path(''),
        transparent=True,
        ftype='svg',
        **kwds):
    """ plot for publication """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sc.pl.umap(adata, color=color, title=title, ax=ax, 
               show=False, **kwds)
    ax.legend_.remove()
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    _save_with_adjust(fig, Path(figdir) / f'umap-{tag}-{color}-{title}.{ftype}', 
                          transparent=transparent, )
    return ax