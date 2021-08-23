# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:22:55 2020

@author: Xingyan Liu
"""
from typing import Union, Sequence, Mapping, Optional
import logging
import numpy as np
import pandas as pd
import scanpy as sc

from scipy import sparse
import torch as th
import dgl

from ..utils import preprocess as utp
from ..utils.base import save_pickle


# In[]

class AlignedDataPair(object):
    """ paired datasets with the aligned features
    
    Parameters
    ----------
    
    features: list or tuple
        a list or tuple of 2 feature matrices.
        common / aligned feratures, as node-features (for observations).
        of shape (n_obs1, n_features) and (n_obs2, n_features)

    ov_adjs: list or tuple
        a list or tuple of 2 (sparse) feature matrices.
        unaligned features, for making `ov_adj`.
        of shape (n_obs1, n_vnodes1) and (n_obs2, n_vnodes2)

    varnames_feat: list or tuple
        names of variables that will be treated as node-features for observations

    varnames_node: list or tuple
        names of variables that will be treated as nodes.

    obs_dfs: list or tuple
        a list or tuple of 2 DataFrame s

    ntypes: dict
    etypes: dict
    **kwds:
        other key words for constructiong of the HeteroGraph
    

    Attributes
    ----------
    
    _features:
    _ov_adjs:
    vv_adj: var-var adjacent matrix (e.g. gene-gene adjacent matrix)
    ov_adj: combined observation-by-variable adjacent matrix (e.g. cell-gene adjacent matrix)
    G: dgl.Heterograph
    n_obs
    n_obs1
    n_obs2
    n_vnodes
    n_vnodes1
    n_vnodes2
    ntypes
    etypes
    dataset_names
    
    
    Examples
    --------

    >>> DataPair([features1, features2],
    ...          [ov_adj1, ov_adj2],
    ...          varnames_feat = vars_feat,
    ...          varnames_node = vars_node,
    ...          obs_dfs = [obs1, obs2],
    ...          dataset_names=dataset_names,
    ...          )

    """
    KEY_DATASET = 'dataset'
    KEY_VARNAME = 'name'
    KEY_OBSNAME = 'original_name'
    DEFAULT_NTYPES = {'o': 'cell', 'v': 'gene'}
    DEFAULT_ETYPES = {'ov': 'express',
                      'vo': 'expressed_by',
                      'oo': 'similar_to'
                      }

    def __init__(
            self,
            features: Sequence[Union[sparse.spmatrix, np.ndarray]],
            ov_adjs: Sequence[Union[sparse.spmatrix, np.ndarray]],
            oo_adjs: Optional[Sequence[sparse.spmatrix]] = None,
            varnames_feat: Optional[Sequence[str]] = None,
            varnames_node: Optional[Sequence[str]] = None,
            obs_dfs: Optional[Sequence[pd.DataFrame]] = None,
            var_dfs: Optional[Sequence[pd.DataFrame]] = None,  # TODO!!!
            dataset_names: Sequence[str] = ('reference', 'query'),
            ntypes: Mapping[str, str] = DEFAULT_NTYPES,
            etypes: Mapping[str, str] = DEFAULT_ETYPES,
            make_graph: bool = True,
            **kwds
    ):

        self.set_dataset_names(dataset_names)

        self.set_features(features, varnames_feat)
        self.set_ov_adj(ov_adjs)
        self.set_oo_adj(oo_adjs)
        self.set_obs_dfs(*obs_dfs)
        self.set_varnames_node(varnames_node)

        self.set_common_obs_annos(ignore_index=True)  # TODO!!!
        #        self.set_vnode_annos(ignore_index=True)
        self.set_ntypes(ntypes)
        self.set_etypes(etypes)
        self.make_whole_net(**kwds)

    def save_init(self, path='datapair_init.pickle'):
        """
        save object for reloading

        Examples
        --------
        >>> adpair.save_init('datapair_init.pickle')
        >>> element_dict = load_pickle('datapair_init.pickle')
        >>> adpair = AlignedDataPair(**element_dict)
        """
        element_dict = dict(
            features=self._features,
            ov_adjs=self._ov_adjs,
            oo_adjs=self._oo_adj,
            varnames_feat=self._varnames_feat,
            varnames_node=self._varnames_node,
            obs_dfs=self.obs_dfs,
            dataset_names=self.dataset_names,
            ntypes=self.ntypes,
            etypes=self.etypes,
        )
        save_pickle(element_dict, path)
        logging.info(f"inputs for construction (aligned) datapair saved into {path}")

    def get_obs_features(self, astensor=True, scale=True,
                         unit_var=True,
                         clip=False, clip_range=(-3, 3.5)):
        feats = self._features
        if scale:
            def zscore(X, with_mean=True, unit_var=unit_var, ):
                return utp.zscore(X, with_mean=with_mean, scale=unit_var)

            feats = list(map(zscore, feats))

        features = np.vstack(feats)
        if clip:
            vmin, vmax = clip_range
            print(f'clipping feature values within [{vmin: .2f}, {vmax: .2f}]')
            features = np.clip(features, vmin, vmax)

        if astensor:
            features = th.FloatTensor(features)
        return features

    def get_feature_dict(
            self,
            astensor: bool = True,
            scale: bool = True,
            unit_var: bool = True,
            **kwds):
        features = self.get_obs_features(astensor=astensor, scale=scale,
                                         unit_var=unit_var, **kwds)
        return {self.ntypes['o']: features}

    def get_whole_net(self, rebuild=False, **kwds):

        if rebuild:
            print('rebuilding the Hetero-graph...')
            self.make_whole_net(**kwds)
        return self.G

    def get_obs_labels(self,
                       keys: Union[str, Sequence[Union[str, None]]],
                       astensor=True,
                       train_use: int = 0,
                       add_unknown_force=False,
                       asint=True,
                       split=False,
                       name_unknown='unknown',
                       categories=None):
        """
        make labels for model training
        
        If `categories` is provided, the labels will be coded according to it,
        and the names not in `categories` will be NA or `name_unknown`('unknown')

        """
        labels_12 = self.get_obs_anno(keys, concat=False)
        classes = list(labels_12[train_use].unique()) if categories is None else list(categories)

        cats_12 = tuple(map(set, labels_12))
        oneshot = len(cats_12[1 - train_use].difference(cats_12[train_use])) >= 1

        if oneshot or add_unknown_force:
            classes += [name_unknown]

        labels_cat = pd.Categorical(list(labels_12[0]) + list(labels_12[1]),
                                    categories=classes)

        if oneshot or add_unknown_force:
            labels_cat = labels_cat.fillna(name_unknown)
        print(labels_cat.value_counts())

        if asint:
            labels = labels_cat.codes.copy()
            if astensor:
                labels = th.LongTensor(labels)
            if split:
                labels = (labels[: self.n_obs1],
                          labels[self.n_obs1: self.n_obs])

            return labels, classes
        else:
            if split:
                labels_cat = (labels_cat[: self.n_obs1],
                              labels_cat[self.n_obs1: self.n_obs])
            return labels_cat

    def get_obs_anno(self,
                     keys: Union[str, Sequence[Union[str, None]]],
                     which=None,
                     concat=True):
        """
        get the annotations of samples (observations)
        """
        # annotations for only one dataset
        if which is not None:
            return self.obs_dfs[which][keys]

        # annotations for both datasets, probably with different keys
        if isinstance(keys, str):
            keys = [keys] * 2
        if keys[0] is None:
            anno1 = [np.nan] * self.n_obs1
        else:
            anno1 = self.obs_dfs[0][keys[0]]

        if keys[1] is None:
            anno2 = [np.nan] * self.n_obs2
        else:
            anno2 = self.obs_dfs[1][keys[1]]

        if concat:
            return np.array(anno1.tolist() + anno2.tolist())
        else:
            return anno1, anno2

    def get_obs_dataset(self, ):
        return self.obs[self.KEY_DATASET]

    def get_obs_ids(self, which=0, astensor=True, ):
        """
        get node indices for obs-nodes (samples), choices are:
            1. all the node ids
            2. only the ids of the "reference" data (by `which=0`)
            3. only the ids of the "query" data (by `which=1`)
        """
        if which is None:
            obs_ids = np.arange(self.n_obs)
        elif which in [0, self.dataset_names[0]]:
            obs_ids = np.arange(self.n_obs1)
        elif which in [1, self.dataset_names[1]]:
            obs_ids = np.arange(self.n_obs1, self.n_obs)
        else:
            raise ValueError(
                '`which` should be one of the following:\n\t'
                f'None, 0, 1, {self.dataset_names[0]}, {self.dataset_names[1]}. '
                f'got {which}')
        if astensor:
            return th.LongTensor(obs_ids)
        else:
            return obs_ids

    @property
    def vnode_ids(self, ):
        return np.arange(self.n_vnodes)

    @property
    def obs_ids1(self, ):
        return self.get_obs_ids(0, False)

    @property
    def obs_ids2(self, ):
        return self.get_obs_ids(1, False)

    @property
    def obs_ids(self, ):
        return self.get_obs_ids(None, False)

    #    def get_vnode_ids_by_name(self, varlist, which = 0, unseen=np.nan):
    #        """
    #        looking-up var-node indices for the given names
    #        """
    #        if isinstance(varlist, str):
    #            varlist = [varlist]
    #        gids = [self._n2i_dict[which].get(g, unseen) for g in varlist]
    #        return gids
    # In[]
    def make_ov_adj(self, ):

        return self._ov_adj

    def make_whole_net(self, selfloop_o=True, selfloop_v=True, ):
        """
        make the whole hetero-graph (e.g. cell-gene graph)
        """
        ntypes = self.ntypes
        etypes = self.etypes
        ov_adj = self.make_ov_adj()

        edge_dict = {
            (ntypes['o'], etypes['ov'], ntypes['v']): ov_adj,
            (ntypes['v'], etypes['vo'], ntypes['o']): ov_adj.T,
        }
        if self._oo_adj is not None:
            edge_dict[(ntypes['o'], etypes['oo'], ntypes['o'])] = self._oo_adj
        if selfloop_o:
            edge_dict[
                (ntypes['o'], f'self_loop_{ntypes["o"]}', ntypes['o'])
            ] = sparse.eye(self.n_obs)
        if selfloop_v:
            edge_dict[
                (ntypes['v'], f'self_loop_{ntypes["v"]}', ntypes['v'])
            ] = sparse.eye(self.n_vnodes)

        # for compatibility with the new version of DGL
        edge_dict = utp.scipy_edge_dict_for_dgl(edge_dict)
        self.G = dgl.heterograph(edge_dict)

        self.info_net = dict(
            edge_types=tuple(edge_dict.keys()),
            selfloop_o=selfloop_o,
            selfloop_v=selfloop_v
        )
        self.summary_graph()

    # In[]
    def set_dataset_names(self, dataset_names):

        if len(dataset_names) == 2:
            print('[*] Setting dataset names:\n\t0-->{}\n\t1-->{}'.format(*dataset_names))
            self.dataset_names = tuple(dataset_names)
        else:
            raise ValueError('`dataset_names` should be of length 2!')

    def set_features(self, features, varnames_feat=None):
        """
        setting feature matrices, where features are aligned across datasets.
        """
        if len(features) == 2:
            print('[*] Setting aligned features for observation nodes '
                  '(self._features)')
            self.n_obs1, n_ft1 = features[0].shape
            self.n_obs2, n_ft2 = features[1].shape

            if n_ft1 == n_ft2:
                self.n_feats = n_ft1
            else:
                raise ValueError(f'The second dimension of the two matrices '
                                 'must be the same ! got {n_ft1} and {n_ft2}')
            self.n_obs = self.n_obs1 + self.n_obs2
            self._features = tuple(features)

            if varnames_feat is None:
                varnames_feat = list(range(self.n_feats))
            #            feats = list(zip(*map(list, varnames_feat)))
            self._varnames_feat = pd.Series(varnames_feat)

        else:
            raise ValueError('`features` should be a list or tuple of length 2!')

    def set_ov_adj(self, ov_adjs):
        """
        set observation-by-variable adjacent matrices
        """
        if len(ov_adjs) == 2:
            print('[*] Setting observation-by-variable adjacent matrices '
                  '(`self._ov_adjs`) for making merged graph adjacent matrix of '
                  'observation and variable nodes')
            n_obs1, n_vnodes1 = ov_adjs[0].shape
            n_obs2, n_vnodes2 = ov_adjs[1].shape
            # checking dimensions
            if n_obs1 != self.n_obs1 or n_obs2 != self.n_obs2:
                raise ValueError('[AlignedDataPair] '
                                 'The first dimensions of the adjacent matrix '
                                 'matrices `ov_adjs` and the common-feature '
                                 'matrices `self._features` are not matched !')

            if n_vnodes1 != n_vnodes2:
                raise ValueError('[AlignedDataPair] '
                                 'The second dimensions of the adjacent matrices '
                                 'matrices `ov_adjs` are not matched! '
                                 f'({n_vnodes1} != {n_vnodes2})')

            def _process_spmat(adj):
                adj = sparse.csr_matrix(adj)
                adj.eliminate_zeros()
                return adj

            self._ov_adjs = tuple(map(_process_spmat, ov_adjs))
            self.n_vnodes = n_vnodes1
        else:
            raise ValueError(f'`ov_adjs` should be a list or tuple of 2 '
                             f'sparse matrices! (got {len(ov_adjs)})')

    @property
    def _ov_adj(self, ):
        """ merged adjacent matrix between observation and variable nodes 
        """
        return sparse.vstack(self._ov_adjs)

    def set_varnames_node(self, varnames_node=None, index=None):
        if varnames_node is None:
            varnames_node = list(range(self.n_vnodes))
        #        self._varnames_node = pd.Series(varnames_node)
        if not hasattr(self, 'var'):
            self.var = pd.DataFrame({'name': varnames_node}, index=index)
        else:
            self.var['name'] = varnames_node

    @property
    def _varnames_node(self, ):
        return self.var['name'].values.copy()

    def set_oo_adj(self, oo_adjs=None):
        if oo_adjs is None:
            self._oo_adj = None
        elif isinstance(oo_adjs, Sequence):
            self._oo_adj = sparse.block_diag(oo_adjs)
        elif sparse.isspmatrix(oo_adjs):
            self._oo_adj = oo_adjs.copy()
        else:
            raise ValueError(
                'if provided, `oo_adjs` should be either a scipy.spmatrix '
                'matrix or a sequence of two sparse matrices')

    def set_obs_dfs(self,
                    obs1: Union[None, pd.DataFrame] = None,
                    obs2: Union[None, pd.DataFrame] = None):

        def _check_obs(obs, n_obs):
            if obs is None:
                obs = pd.DataFrame(index=range(n_obs))
            elif obs.shape[0] != n_obs:
                raise ValueError(f'the number of observations are not matched '
                                 f'expect {n_obs}, got {obs.shape[0]}.')
            print(obs.columns)
            return obs

        obs1 = _check_obs(obs1, self.n_obs1, )  # self.dataset_names[0]
        obs2 = _check_obs(obs2, self.n_obs2, )  # self.dataset_names[1]
        self.obs_dfs = [obs1, obs2]
        self._obs_id2name = pd.Series(obs1.index.tolist() + obs2.index.tolist())

    def set_common_obs_annos(self,
                             df: Union[None, pd.DataFrame] = None,
                             ignore_index=True,
                             **kwannos):
        """
        Shared and merged annotation labels for ALL of the observations in both
        datasets. (self.obs, pd.DataFrame)
        """
        if not hasattr(self, 'obs'):
            self.obs = self._obs_id2name.to_frame(self.KEY_OBSNAME)
            dsn_lbs = self.n_obs1 * [self.dataset_names[0]] + \
                      self.n_obs2 * [self.dataset_names[1]]
            self.obs[self.KEY_DATASET] = pd.Categorical(
                dsn_lbs, categories=self.dataset_names)

        self._set_annos(self.obs, df,
                        ignore_index=ignore_index,
                        copy=False, **kwannos)

    def set_ntypes(self, ntypes: Mapping[str, str]):
        if utp.dict_has_keys(ntypes, 'o', 'v'):
            self.ntypes = ntypes
        else:
            raise KeyError('the dict for `ntypes` should have 2 keys: '
                           '"o" (for observation types) and "v" (for variable types)')

    def set_etypes(self, etypes: Mapping[str, str]):
        if utp.dict_has_keys(etypes, 'ov', 'vo'):
            self.etypes = etypes
        else:
            raise KeyError('the dict for `etypes` should have 2 keys:',
                           '"ov" (for observation-variable edge type)',
                           '"vo" (for variable-observation edge type)')

    def summary_graph(self, ):
        if hasattr(self, 'G') and hasattr(self, 'info_net'):
            info = self.info_net

            print('-' * 20, 'Summary of the DGL-Heterograph', '-' * 20)
            print(self.G)
            print('self-loops for observation-nodes: {}'.format(info['selfloop_o']))
            print('self-loops for variable-nodes: {}'.format(info['selfloop_v']))
        else:
            print("graph haven't been made, run `self.make_whole_net(...)` first!")

    @staticmethod
    def _set_annos(df0, df=None, ignore_index=True,
                   copy=False, **kwannos):
        if copy: df0 = df0.copy()
        if df is not None:
            for col in df.columns:
                df0[col] = df[col].tolist() if ignore_index else df[col]

        if len(kwannos) >= 1:
            for k, v in kwannos.items():
                # be careful of the index mismatching problems!!!
                df0[k] = list(v) if ignore_index else v
        return df0 if copy else None


# In[]
"""
"""


def _check_sparse_toarray(mat):
    if sparse.issparse(mat):
        mat = mat.toarray()
    return mat


def _check_array_tosparse(mat, scilent=True):
    if not sparse.issparse(mat):
        if not scilent:
            print('transforming dense matrix to CSR sprse matrix')
        mat = sparse.csr_matrix(mat)
    return mat


def aligned_datapair_from_adatas(
        adatas: Sequence[sc.AnnData],
        vars_feat: Sequence,
        vars_as_nodes: Optional[Sequence] = None,
        oo_adjs: Optional[Sequence[sparse.spmatrix]] = None,
        dataset_names: Sequence[str] = ('reference', 'query'),
        **kwds
) -> AlignedDataPair:
    """
    Build ``AlignedDataPair`` object from a pair of adatas

    Parameters
    ----------
    
    adatas:
        a list or tuple of 2 sc.AnnData object.
    
    vars_feat: a sequence of variable names
        a name-list of variables that will be used as (cell) node features.
        for example, names of differentail expressed genes (DEGs),
        highly variable features.
    
    vars_as_nodes: a sequence of variable names, optional.
        a name-list of variables that will be taken as nodes in the graph
        for model training.
        if None (not provided), it will be the same as `vars_feat`
    
    oo_adjs:
        a sequence of (sparse) adjacent matrices of observations.
        for example, the single-cell network within each dataset.

    dataset_names:
        dataset names, for example, ('ref', 'que')

    Returns
    -------
    dpair: AlignedDataPair


    Examples
    --------
    >>> dpair = datapair_from_adatas(
    ...     [adata1, adata2],
    ...     vars_feat,
    ...     dataset_names = ['reference', 'query']
    ...     )

    """
    adata1, adata2 = adatas
    adata_raw1 = adata1.raw.to_adata() if adata1.raw is not None else adata1
    adata_raw2 = adata2.raw.to_adata() if adata2.raw is not None else adata2

    vars_common = set(adata_raw1.var_names).intersection(adata_raw2.var_names)
    # features selected for modeling. (e.g. DEGs, HVGs)
    vars_feat = list(vars_common.intersection(vars_feat))
    vars_as_nodes = vars_feat if vars_as_nodes is None else list(
        vars_common.intersection(vars_as_nodes))

    # --- obs. annotation dataframes
    obs1 = adata1.obs.copy()
    obs2 = adata2.obs.copy()

    # --- node features (for single-cells)
    # try:
    #     features1 = adata1[:, vars_feat].X
    #     features2 = adata2[:, vars_feat].X
    # except:
    logging.info(
        '[NOTE]\nthe node features will be extracted from `adata.raw`, '
        'please make sure that the values are normalized.\n')
    features1 = adata_raw1[:, vars_feat].X
    features2 = adata_raw2[:, vars_feat].X

    features = list(map(_check_sparse_toarray, [features1, features2]))

    # --- ov_adjs (features for making `ov_adj`)
    ov_adj1 = adata_raw1[:, vars_as_nodes].X  # for ov_adj
    ov_adj2 = adata_raw2[:, vars_as_nodes].X

    return AlignedDataPair(
        features,
        [ov_adj1, ov_adj2],
        oo_adjs=oo_adjs,
        varnames_feat=vars_feat,
        varnames_node=vars_as_nodes,
        obs_dfs=[obs1, obs2],
        dataset_names=dataset_names,
        **kwds)
