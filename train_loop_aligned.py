# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:57:36 2021

@author: Administrator
"""

import scanpy as sc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.chdir('D:/lxy/scr4sent-20210315')

sys.path.append('.')

import CAME
from CAME.utils.base import make_pairs_from_lists
from CAME import pipeline, pp, pl

from DATASET_NAMES import NAMES_ALL, Tissues
pipeline.seed_everything()


# In[]

# datadir = Path('D:/SQL/datasets')
# dir_gmap = Path('D:/lxy/resources/exported_gene_matches')

datadir = Path('../datasets')
dir_gmap = Path('../resources/mart_exports/exported_gene_matches')

RESDIR = Path('./results')

key_class = 'cell_ontology_class'
key_clust = 'clust_lbs'
header = 'scanpy'


def foo_rmv(args):
    adt, key_class = args
    if key_class in adt.obs.columns:
        return pp.remove_adata_small_groups(
                adt, key_class, min_samples=10)
    else:
        print(f'"{key_class}" is not in the obs.columns, skipped')
        return adt


# In[]
# for parameter...
params_model = CAME.get_model_params()
params_lossfunc = CAME.get_loss_params()

only_1v1homo = False
use_scnets = True
n_epochs = 400
n_pass = 100
batch_size = 32

resdir0 = RESDIR / 'minibatch' / f'batch_size-{batch_size}'





# In[]
for tiss in Tissues:
    print(f'Tissue: {tiss}')
    dir_formal = datadir / 'formal' / tiss
    NameDict = NAMES_ALL[tiss]
    species = list(NameDict.keys())
    pair_species = make_pairs_from_lists(species, species)
    #sp1, sp2 = 'human', 'mouse'
    for sp1, sp2 in pair_species:

        df_varmap_1v1 = pd.read_csv(dir_gmap / f'gene_matches_1v1_{sp1}2{sp2}.csv',)

        pair_datasets = make_pairs_from_lists(NameDict[sp1], NameDict[sp2])
        ####################################################
        for dsnames in pair_datasets:
            dsn1, dsn2 = dsnames
            if dsn2 == 'merged_mouse':
                continue
            dsnames = (dsn1, dsn2)
#            if dataset_names not in DATASET_PAIRS:
#                continue

            resdir = resdir0 / '---'.join([header, dsn1, dsn2])
            if (resdir / 'info.txt').exists():
                print(f'already run for {dsnames}, skipped')
                continue

            print(f'Step 0: loading adatas {dsnames}...')
            fn1 = dir_formal / f'raw-{dsn1}.h5ad'
            fn2 = dir_formal / f'raw-{dsn2}.h5ad'

            adatas = sc.read_h5ad(fn1), sc.read_h5ad(fn2)
            # remove rare
            # adatas = tuple(map(foo_rmv, zip(adatas, [key_class] * 2)))

            print('================ preprocessing ===============')
            came_inputs, (adata1, adata2) = pipeline.preprocess_aligned(
                adatas,
                key_class=key_class,
                df_varmap_1v1=df_varmap_1v1,
            )

            _ = pipeline.main_for_aligned(
                **came_inputs,
                dataset_names=dsnames,
                key_class1=key_class,
                key_class2=key_class,
                do_normalize=True,
                n_epochs=n_epochs,
                resdir=resdir,
                check_umap=not True,
                # True for visualizing embeddings each 40 epochs
                n_pass=100,
                params_model=dict(residual=False),
                plot_results=True, # TODO: if raise error, change it to False
            )

    # In[]
""" Gather results from RGCN """


def values_by_best_index(
        logdf: pd.DataFrame,
        keys='test_acc',
        baseon='AMI',
        index_label='epoch',
        nepoch_pass=0,
        dictout=True
):
    # note that the index numbers are not changed
    logdf = logdf.iloc[nepoch_pass:, :]
    indx = logdf[baseon]
    iepoch = indx.idxmax()
    if isinstance(keys, str):
        keys_out = [keys, baseon]
    else:
        keys_out = [k for k in list(keys) + [baseon] if k in logdf.columns]

    if dictout:
        dct = logdf.loc[iepoch, keys_out].to_dict()
        dct[index_label] = iepoch
        return dct
    else:
        vals = logdf.loc[iepoch, keys_out]
        return iepoch, vals


def record_from_logdf(
        logdf,
        key_acc=['test_acc', 'microF1', 'macroF1', 'weightedF1'],
        baseon='AMI',
        nepoch_pass=50
):
    dct = values_by_best_index(
        logdf, keys=key_acc, baseon=baseon, nepoch_pass=nepoch_pass)

    iep, vals = values_by_best_index(
        logdf, keys=baseon, baseon='test_acc', dictout=False)
    ami_best, acc_best = vals

    dct.update(acc_best=acc_best, NMI_best=ami_best, epoch_best=iep)
    return dct


dir_restb = resdir
baseon = 'AMI'
nepoch_pass = n_pass  # 100

if True:
    records = {}
    subdirs = [d for d in os.listdir(dir_restb) if d.startswith(header)]
    for _subd in subdirs[:]:

        _, _dsn1, _dsn2 = _subd.split('---')
        _resdir = dir_restb / _subd
        if not (_resdir / 'train_logs.csv').exists():
            continue
        logdf = pd.read_csv(_resdir / 'train_logs.csv', index_col=0)
        # select and recording accuracies
        dct = record_from_logdf(logdf, baseon=baseon, nepoch_pass=nepoch_pass)
        records[(_dsn1, _dsn2)] = dct

    if len(records) >= 1:
        df_records = pd.DataFrame(records).T
        print(df_records[['test_acc', 'acc_best']].describe())
        print(df_records.head())
        df_records.to_csv(dir_restb / f'metrics-after{nepoch_pass}.csv',
                          index_label=('reference', 'query'))

