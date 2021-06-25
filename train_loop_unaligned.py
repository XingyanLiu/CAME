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
RESDIR = Path('./results')

# datadir = Path('D:/SQL/datasets')
# dir_gmap = Path('D:/lxy/resources/exported_gene_matches')

datadir = Path('datasets')
dir_gmap = Path('resources/mart_exports/exported_gene_matches')

key_class = 'cell_ontology_class'
key_clust = 'clust_lbs'
header = 'scanpy'


# In[]
# for parameter...
params_model = CAME.get_model_params()
params_lossfunc = CAME.get_loss_params()

only_1v1homo = False
use_scnets = True
n_epochs = 400
n_pass = 100
batch_size = 32

resdir = RESDIR / 'minibatch' / f'batch_size-{batch_size}'

for tiss in Tissues:
    print(f'Tissue: {tiss}')
    dir_formal = datadir / 'formal' / tiss
    NameDict = NAMES_ALL[tiss]
    species = list(NameDict.keys())
    pair_species = make_pairs_from_lists(species, species)
    # sp1, sp2 = 'human', 'mouse'
    for sp1, sp2 in pair_species:

        df_varmap_1v1 = pd.read_csv(dir_gmap / f'gene_matches_1v1_{sp1}2{sp2}.csv',)
        if only_1v1homo:
            df_varmap = df_varmap_1v1
        else:
            df_varmap = pd.read_csv(dir_gmap / f'gene_matches_{sp1}2{sp2}.csv',)

        pair_datasets = make_pairs_from_lists(NameDict[sp1], NameDict[sp2])
        ####################################################
        for dsnames in pair_datasets:
            dsn1, dsn2 = dsnames
            if dsn2 == 'merged_mouse':
                continue

            fdir = resdir / '---'.join([header, dsn1, dsn2])
            if fdir.exists():
                print(f'already run for {dsnames}, skipped')
                continue

            print(f'Step 0: loading adatas {dsnames}...')

            fn1 = dir_formal / f'raw-{dsn1}.h5ad'
            fn2 = dir_formal / f'raw-{dsn2}.h5ad'

            adata_raw1, adata_raw2 = sc.read_h5ad(fn1), sc.read_h5ad(fn2)
            adatas = [adata_raw1, adata_raw2]

            print('================ step1: preprocessing ===============')
            came_inputs, (adata1, adata2) = pipeline.preprocess_unaligned(
                    adatas,
                    key_class=key_class,
                    use_scnets=use_scnets
                )
            _ = pipeline.main_for_unaligned(
                    **came_inputs,
                    df_varmap=df_varmap,
                    df_varmap_1v1=df_varmap_1v1,
                    dataset_names=dsnames,
                    key_class1=key_class,
                    key_class2=key_class,
                    do_normalize=True,
                    n_epochs=n_epochs,
                    resdir=resdir,
                    check_umap=not True,
                    n_pass=100,
                    params_model=params_model,
                    params_lossfunc=params_lossfunc,
                    batch_size=batch_size,
                )

            del _

            torch.cuda.empty_cache()
            print('memory cleared\n')

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
nepoch_pass = n_pass # 100

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


