# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:57:36 2021

@author: Administrator
"""

import scanpy as sc
import os
from pathlib import Path


import numpy as np
import pandas as pd
#from scipy import sparse
import matplotlib as mpl
mpl.use('Agg')
#import networkx as nx
import torch

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.chdir('D:/lxy/scr4sent-20210315')
os.chdir('E:/lxy_pro/004/main')

from CAME_v0 import PARAMETERS
from CAME_v0.utils import preprocess as utp
from CAME_v0.utils import base
from DATASET_NAMES import NAMES_ALL, Tissues

import packed_pipeline 
packed_pipeline.seed_everything()


# In[]
RESDIR = Path('./results')

datadir = Path('D:/SQL/datasets')
dir_gmap = Path('D:/lxy/resources/exported_gene_matches')

datadir = Path('datasets')
dir_gmap = Path('resources/mart_exports/exported_gene_matches')

key_class = 'cell_ontology_class'
key_clust = 'clust_lbs'
hvg_source = 'scanpy'

memory_restricted = False


# In[]
#for parameter...
params_pre = PARAMETERS.params_pre,
params_model = PARAMETERS.params_model,
params_lossfunc = PARAMETERS.params_lossfunc,

use_scnets = True

for rate in [0.75, 0.5, 0.25, 0.1]:
    
    sample_which = ['ref', 'que', 'refque'][0]
    tag = f'dcounts{rate:.2g}'
    resdir = RESDIR / 'dsample_counts' / f'{sample_which}-{tag}'
    
    utp.check_dirs(resdir)
    base.write_info(
            resdir / 'parameters.txt',
            use_scnets = use_scnets,
            params_pre = params_pre,
            params_model = params_model,
            params_lossfunc = params_lossfunc,
            )
            
    
    
    # In[]
    #tiss = Tissues[0]
    for tiss in Tissues: 
        print(f'Tissue: {tiss}')
        dir_formal = datadir / 'formal' / tiss 
        NameDict = NAMES_ALL[tiss]
        species = list(NameDict.keys())
        pair_species = utp.make_pairs_from_lists(species, species)
        #sp1, sp2 = 'human', 'mouse'
        for sp1, sp2 in pair_species:
            
            df_varmap_1v1 = pd.read_csv(dir_gmap / f'gene_matches_1v1_{sp1}2{sp2}.csv',)
            if params_pre['only_1v1homo']:
                df_varmap = df_varmap_1v1
            else:
                df_varmap = pd.read_csv(dir_gmap / f'gene_matches_{sp1}2{sp2}.csv',)
            
            pair_datasets = utp.make_pairs_from_lists(NameDict[sp1], NameDict[sp2])
            ####################################################
            for name_data1, name_data2 in pair_datasets: # name_data1, name_data2 = pair_datasets[0]
                if name_data2 == 'merged_mouse':
                    continue
                dataset_names = (name_data1, name_data2)
    #            if dataset_names not in DATASET_PAIRS:
    #                continue
                
                fdir = resdir / '---'.join([hvg_source, name_data1, name_data2])
                if fdir.exists():
                    print(f'already run for {dataset_names}, skipped')
                    continue
                
                print(f'Step 0: loading adatas {dataset_names}...')
                if 'ref' in sample_which:
                    fn1 = dir_formal / 'counts_downsample' / f'{tag}-{name_data1}.h5ad'
                else:
                    fn1 = dir_formal / f'raw-{name_data1}.h5ad'
                if 'que' in sample_which:
                    fn2 = dir_formal / 'counts_downsample' / f'{tag}-{name_data2}.h5ad'
                else:
                    fn2 = dir_formal / f'raw-{name_data2}.h5ad'
                
                adata_raw1, adata_raw2 = sc.read_h5ad(fn1), sc.read_h5ad(fn2)
    
               
                print('================ step1: preprocessing ===============')
                params_preproc = dict(
                        target_sum = None,
                        n_top_genes = 2000,
                        nneigh = 5,)
                #NOTE: using the median total-counts as the scale factor (better than fixed number)
                adata1 = utp.quick_preprocess(adata_raw1, **params_preproc)
                adata2 = utp.quick_preprocess(adata_raw2, **params_preproc)
                
                # the single-cell network
                if use_scnets:
                    scnets = [utp.get_scnet(adata1), utp.get_scnet(adata2)]
                else:
                    scnets = None
                    
                # get HVGs
                hvgs1 = utp.get_hvgs(adata1, )
                hvgs2 = utp.get_hvgs(adata2, )

                ''' cluster labels '''

                nneigh_clust = 20 
                key_clust = 'clust_lbs'
                clust_lbs2 = utp.get_leiden_labels(
                    adata2, force_redo=True, 
                    nneigh=nneigh_clust, 
                    neighbors_key='clust', 
                    key_added=key_clust,
                    copy=False
                    )    
                adata_raw2.obs[key_clust] = clust_lbs2
    
    
                ####################################################
                print('Step 2: training & recording')
                adatas = [adata_raw1, adata_raw2]
                
                ntop_deg = 50
                params_deg = dict(n=ntop_deg, force_redo=False, 
                                  inplace=True, do_normalize=False)
                ### need to be normalized first
                degs1 = utp.compute_and_get_DEGs(
                        adata1, key_class, **params_deg)
                degs2 = utp.compute_and_get_DEGs(
                        adata2, 'clust_lbs', **params_deg)
                ###
                vars_use = [degs1, degs2]
                vars_as_nodes = [np.unique(np.hstack([hvgs1, degs1])), 
                                 np.unique(np.hstack([hvgs2, degs2]))]
                union_node_feats = True

                
                n_epochs = 400
                res, trainer, ENV_VARs = packed_pipeline.main_without_analysis(
                        adatas,
                        vars_use,
                        vars_as_nodes = vars_as_nodes,
                        dataset_names = dsnames,
                        key_class1 = key_class,
                        key_class2 = key_class,
                        n_epochs = n_epochs,
                        resdir = resdir,
                #        tag_data = f'{tiss}-{dataset_names}-{hvg_source}',
                        params_pre = params_pre, 
                        params_model = params_model, 
                        params_lossfunc=params_lossfunc,
                        check_umap=not True, #True for visualizing embeddings each 40 epochs
                        scnets=scnets,
                        n_pass = 100,
                        )
    
                del res, trainer, ENV_VARs
                torch.cuda.empty_cache()
                print('memory cleared\n')
                
                
                
     # In[]   
    ''' Gather results from RGCN
    '''
    def values_by_best_index(logdf: pd.DataFrame, 
                            keys='test_acc', baseon = 'AMI',
                            index_label = 'epoch',
                            nepoch_pass = 0, 
                            dictout=True):
        logdf = logdf.iloc[nepoch_pass: , :] # note that the index numbers are not changed
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
    
    def record_from_logdf(logdf, 
                          key_acc = ['test_acc', 'microF1', 'macroF1', 'weightedF1'],
                          key_ami = 'AMI',
                          nepoch_pass = 50):
        dct = values_by_best_index(
                logdf, keys=key_acc, baseon = key_ami, nepoch_pass = nepoch_pass)
        
        iep, vals = values_by_best_index(
                logdf, keys=key_ami, baseon = 'test_acc', dictout=False)
        ami_best, acc_best = vals
        
        dct.update(acc_best = acc_best, NMI_best = ami_best, epoch_best=iep)
        return dct
    
    dir_restb = resdir
    key_ami = 'AMI'
    nepoch_pass = 100
    #    subdirs = os.listdir(DIR_RESTB_GNN)
    #    #sub = subdirs[1]
    #    for sub in subdirs:
    #        dir_restb = DIR_RESTB_GNN / f'{sub}'
        
    if True:
        records = {}
        subdirs = [d for d in os.listdir(dir_restb) if d.startswith(hvg_source)]
        for _subd in subdirs[:]:
    
            _, _dsn1, _dsn2 = _subd.split('---')
            _resdir = dir_restb / _subd
            if not (_resdir / 'train_logs.csv').exists():
                continue
            logdf = pd.read_csv(_resdir / 'train_logs.csv', index_col=0)
            logdf.columns
            ### select and recording Accs
            dct = record_from_logdf(logdf, key_ami = key_ami, nepoch_pass=nepoch_pass)
            records[(_dsn1, _dsn2)] = dct
        
        if len(records) >= 1:
            df_records = pd.DataFrame(records).T
            print(df_records[['test_acc', 'acc_best']].describe())
            print(df_records.head())
            df_records.to_csv(dir_restb / f'mtc-GNN-{hvg_source}-after{nepoch_pass}.csv', 
                                index_label=('reference', 'query'))


