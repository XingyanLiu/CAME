# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 23:57:02 2020

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

class DataPair(object):
    """ paired datasets with the un-aligned features
    
    Parameters
    ----------
    
    features: list or tuple
        a list or tuple of 2 feature matrices.
        common / aligned features, as node-features (for observations).
        shape: (n_obs1, n_features) and (n_obs2, n_features)
        
    ov_adjs: list or tuple
        a list or tuple of 2 (sparse) feature matrices.
        unaligned features, for making `ov_adj`.
        of shape (n_obs1, n_vnodes1) and (n_obs2, n_vnodes2)
                
    vv_adj: scipy.sparse COO matrix.
        adjacent matrix between variables from these 2 datasets. 
        (e.g. gene-gene adjacent matrix)
        of shape (n_vnodes, n_vnodes), where `n_vnodes = n_vnodes1 + n_vnodes2`
        is the total number of variable-nodes.
        
    varnames_node: list or tuple
        a list or tuple of 2 name-lists, or one concatenated name-list.
        lengths should be `n_vnodes1` and `v_nodes2`.
        
    obs_dfs: list or tuple
        a list or tuple of 2 `pd.DataFrame`s
    ntypes: dict
    etypes: dict
    
    **kwds:
        other key words for constructiong of the HeteroGraph
    

    Attributes
    ----------
    _features:
    _ov_adjs:
    vv_adj:
        var-var adjacent matrix (e.g. gene-gene adjacent matrix)
    ov_adj:
        observation-by-variable adjacent matrix (e.g. cell-gene adjacent matrix)
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
    ...          vv_adj = vv_adj,
    ...          varnames_node = [vnodes1, vnodes2],
    ...          obs_dfs = [df1, df2],
    ...          dataset_names = ['reference', 'query'],
    ...          )
    
    """
    KEY_DATASET = 'dataset'  # '_datapair_name'
    KEY_VARNAME = 'name'
    KEY_OBSNAME = 'original_name'
    DEFAULT_NTYPES = {'o': 'cell', 'v': 'gene'}
    DEFAULT_ETYPES = {'ov': 'express',
                      'vo': 'expressed_by',
                      'vv': 'homolog_with',
                      'oo': 'similar_to'
                      }

    def __init__(
            self,
            features: Sequence[Union[sparse.spmatrix, np.ndarray]],
            ov_adjs: Sequence[Union[sparse.spmatrix, np.ndarray]],
            vv_adj: sparse.spmatrix,
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
        self.set_vv_adj(vv_adj, varnames_node)
        self.set_oo_adj(oo_adjs)
        self.set_obs_dfs(*obs_dfs)
        # self.set_var_dfs(*var_dfs)  # TODO!!!
        self.set_common_obs_annos(ignore_index=True)
        self.set_vnode_annos(ignore_index=True)
        self.set_ntypes(ntypes)
        self.set_etypes(etypes)
        if make_graph:
            self.make_whole_net(**kwds)
        else:
            print('graph has not been made, call `self.make_whole_net()` if needed.')

    def save_init(self, path='datapair_init.pickle'):
        """
        save object for reloading

        Examples
        --------
        >>> adpair.save_init('datapair_init.pickle')
        >>> element_dict = load_pickle('datapair_init.pickle')
        >>> adpair = DataPair(**element_dict)
        """
        element_dict = dict(
            features=self._features,
            ov_adjs=self._ov_adjs,
            vv_adj=self._vv_adj,
            oo_adjs=self._oo_adj,
            varnames_feat=self._varnames_feat,
            varnames_node=(self.vnode_names1, self.vnode_names2),
            obs_dfs=self.obs_dfs,
            dataset_names=self.dataset_names,
            ntypes=self.ntypes,
            etypes=self.etypes,
        )
        save_pickle(element_dict, path)
        logging.info(f"inputs for construction datapair saved into {path}")

    # In[]
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

    def get_feature_dict(self, astensor=True, scale=False, unit_var=True, **kwds):
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
                       categories=None,
                       set_attr=True,
                       ):
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

        if set_attr:
            self._labels = labels_cat.codes.copy()
            self._classes = classes.copy()

        if asint:
            labels = self.labels.copy()
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

    @property
    def labels(self, ):
        return self._labels.copy() if hasattr(self, '_labels') else None

    @property
    def classes(self, ):
        return self._classes.copy() if hasattr(self, '_classes') else None

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

    #        return self.get_obs_anno(self.KEY_DATASET)

    def get_obs_ids(self, which=0, astensor=True, ):
        """
        get node indices for obs-nodes (samples), choices are:
            1. all the node ids (by `which=None`)
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

    def get_vnode_ids(self, which=0, astensor=True, ):
        """
        get node indices for var-nodes, choices are:
            1. all the node ids
            2. only the ids of the "reference" data (by `which=0`)
            3. only the ids of the "query" data (by `which=1`)
        """
        if which in (None, 'all'):
            vnode_ids = np.arange(self.n_vnodes)
        elif which in (0, self.dataset_names[0]):
            vnode_ids = np.arange(self.n_vnodes1)
        elif which in (1, self.dataset_names[1]):
            vnode_ids = np.arange(self.n_vnodes1, self.n_vnodes)
        else:
            raise ValueError(
                '`which` should be one of the following:\n\t'
                f'None, 0, 1, {self.dataset_names[0]}, {self.dataset_names[1]}. '
                f'got {which}')
        if astensor:
            return th.LongTensor(vnode_ids)
        else:
            return vnode_ids

    def get_vnode_ids_by_name(self, varlist, which=0, unseen=np.nan):
        """
        looking-up var-node indices for the given names
        """
        if isinstance(varlist, str):
            varlist = [varlist]
        gids = [self._n2i_dict[which].get(g, unseen) for g in varlist]
        return gids

    def get_vnode_names(self, vnode_ids=None, tolist=True):

        if vnode_ids is None:
            names = self._var_id2name.copy()
        else:
            names = self._var_id2name[vnode_ids].copy()

        return names.tolist() if tolist else names

    @property
    def vnode_names1(self, ):
        return self._var_id2name[: self.n_vnodes1].tolist()

    @property
    def vnode_names2(self, ):
        return self._var_id2name[self.n_vnodes1:].tolist()

    @property
    def obs_ids1(self, ):
        return self.get_obs_ids(0, False)

    @property
    def obs_ids2(self, ):
        return self.get_obs_ids(1, False)

    @property
    def obs_ids(self, ):
        return self.get_obs_ids(None, False)

    def make_ov_adj(self, link2ord=False):
        """ observation-variable bipartite network
        """
        ov_adj = self._ov_adj.copy()
        if link2ord:
            print('computing the second-order neighbors')
            return ov_adj + ov_adj.dot(self._vv_adj.T)
        return ov_adj

    def make_whole_net(self, link2ord=False, selfloop_o=True, selfloop_v=True,
                       ):
        """
        make the whole hetero-graph (e.g. cell-gene graph)
        """
        ntypes = self.ntypes
        etypes = self.etypes
        ov_adj = self.make_ov_adj(link2ord)
        vv_adj = self._vv_adj.copy()

        edge_dict = {
            (ntypes['o'], etypes['ov'], ntypes['v']): ov_adj,
            (ntypes['v'], etypes['vo'], ntypes['o']): ov_adj.T,
            (ntypes['v'], etypes['vv'], ntypes['v']): vv_adj,
        }
        if self._oo_adj is not None:
            edge_dict[(ntypes['o'], etypes['oo'], ntypes['o'])] = self._oo_adj

        if selfloop_o:
            edge_dict[
                (ntypes['o'], f'self_loop_{ntypes["o"]}', ntypes['o'])
            ] = sparse.eye(self.n_obs)
        if selfloop_v:
            edge_dict[
                (ntypes['v'], etypes['vv'], ntypes['v'])
            ] += sparse.eye(self.n_vnodes)
        #        print('TEST:', edge_dict)
        ### for compatibility with the new version of DGL
        edge_dict = utp.scipy_edge_dict_for_dgl(edge_dict, foo=th.LongTensor)
        self.G = dgl.heterograph(edge_dict, )

        self.info_net = dict(
            edge_types=tuple(edge_dict.keys()),
            link2ord=link2ord,
            selfloop_o=selfloop_o,
            selfloop_v=selfloop_v
        )
        self.summary_graph()

    def set_dataset_names(self, dataset_names: Sequence[str]):

        if len(dataset_names) == 2:
            print('[*] Setting dataset names:\n\t0-->{}\n\t1-->{}'.format(*dataset_names))
            self.dataset_names = tuple(dataset_names)
        else:
            raise ValueError('`dataset_names` should be of length 2!')

    def set_features(self, features, varnames_feat=None):
        """
        setting feature matrices, where features are aligned across datasets.
        varnames_feat:
            if provided, should be a sequence of two name-lists
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
                                 f'must be the same ! got {n_ft1} and {n_ft2}')
            self.n_obs = self.n_obs1 + self.n_obs2
            self._features = tuple(features)

            if varnames_feat is None:
                varnames_feat = [list(range(self.n_feats)), list(range(self.n_feats))]
            feats = list(zip(*map(list, varnames_feat)))
            self._varnames_feat = pd.DataFrame(feats, columns=self.dataset_names)
        else:
            raise ValueError('`features` should be a list or tuple of length 2!')

    def set_ov_adj(self, ov_adjs):
        """
        set un-aligned features, for making observation-variable adjacent matrix
        """
        if len(ov_adjs) == 2:
            print('[*] Setting un-aligned features (`self._ov_adjs`) for '
                  'making links connecting observation and variable nodes')
            n_obs1, self.n_vnodes1 = ov_adjs[0].shape
            n_obs2, self.n_vnodes2 = ov_adjs[1].shape
            if n_obs1 != self.n_obs1 or n_obs2 != self.n_obs2:
                raise ValueError(f'[DataPair] '
                                 'The first dimensions of the unaligned-feature '
                                 'matrices `ov_adjs` and the common-feature '
                                 'matrices `self._features` are not matched !')

            def _process_spmat(adj):
                adj = sparse.csr_matrix(adj)
                adj.eliminate_zeros()
                return adj

            self._ov_adjs = tuple(map(_process_spmat, ov_adjs))
            self.n_vnodes = self.n_vnodes1 + self.n_vnodes2
        else:
            raise ValueError('`ov_adjs` should be a list or tuple of length 2!')

    @property
    def _ov_adj(self, ):
        """ merged adjacent matrix between observation and variable nodes 
        """
        return sparse.block_diag(self._ov_adjs)

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

    def set_vv_adj(self, vv_adj, varnames_node=None):
        """
        vv_adj: 
            adjacent matrix between variables from these 2 datasets.
            e.g. gene-gene network, with each edge connecting each pair of 
            homologous genes.
        """
        print('[*] Setting adjacent matrix connecting variables from these '
              '2 datasets (`self._vv_adj`)')
        self._vv_adj = sparse.csr_matrix(vv_adj)
        self._vv_adj.eliminate_zeros()
        if varnames_node is None:
            varnames_node = (np.arange(self.n_vnodes1),
                             np.arange(self.n_vnodes1, self.n_vnodes))
        self._var_id2name, self._n2i_dict = utp.make_id_name_maps(*varnames_node)

    def set_var_dfs(self, var1, var2):
        pass  # TODO! maybe not necessary

    def set_obs_dfs(self,
                    obs1: Union[None, pd.DataFrame] = None,
                    obs2: Union[None, pd.DataFrame] = None):
        """
        private observation annotaions
        """

        def _check_obs(obs, n_obs, ):  # val, key = self.KEY_DATASET
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

    def set_vnode_annos(self,
                        df: Union[None, pd.DataFrame] = None,
                        ignore_index=True,
                        force_reset=False,
                        **kwannos):

        if not hasattr(self, 'var') or force_reset:
            self.var = self._var_id2name.to_frame(self.KEY_VARNAME)
            dsn_lbs = self.n_vnodes1 * [self.dataset_names[0]] + \
                      self.n_vnodes2 * [self.dataset_names[1]]
            self.var[self.KEY_DATASET] = pd.Categorical(
                dsn_lbs, categories=self.dataset_names)
            # inferring variable-nodes that have homologues in the other dataset
            n_corresponds = self._vv_adj.sum(1).A1
            self.var['is_linked'] = pd.Categorical(n_corresponds >= 1)
            self.var['is_linked_1v1'] = pd.Categorical(n_corresponds == 1)

        self._set_annos(self.var, df,
                        ignore_index=ignore_index,
                        copy=False, **kwannos)

    def set_ntypes(self, ntypes: Mapping[str, str]):
        if utp.dict_has_keys(ntypes, 'o', 'v'):
            self.ntypes = ntypes
        else:
            raise KeyError('the dict for `ntypes` should have 2 keys: '
                           '"o" (for observation types) and "v" (for variable types)')

    def set_etypes(self, etypes: Mapping[str, str]):
        if utp.dict_has_keys(etypes, 'ov', 'vo', 'vv'):
            self.etypes = etypes
        else:
            raise KeyError('the dict for `etypes` should have 3 keys:',
                           '"ov" (for observation-variable edge type)',
                           '"vo" (for variable-observation edge type)',
                           '"vv" (for edge type between variables)')

    def summary_graph(self, ):
        if hasattr(self, 'G') and hasattr(self, 'info_net'):
            info = self.info_net

            print('-' * 20, 'Summary of the DGL-Heterograph', '-' * 20)
            print(self.G)
            print('second-order connection: {}'.format(info['link2ord']))
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
# functions for `DataPair` object construction from `sc.AnnData`


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


def datapair_from_adatas(
        adatas: Sequence[sc.AnnData],
        vars_use: Sequence[Sequence],
        df_varmap: pd.DataFrame,
        df_varmap_1v1: Optional[pd.DataFrame] = None,
        oo_adjs: Optional[Sequence[sparse.spmatrix]] = None,
        vars_as_nodes: Union[None, Sequence[Sequence]] = None,
        union_node_feats: Union[str, bool] = 'auto',
        dataset_names: Sequence[str] = ('reference', 'query'),
        with_single_vnodes: bool = True,
        **kwds
) -> DataPair:
    """
    Build ``DataPair`` object from a pair of adatas

    Parameters
    ----------

    adatas: list or tuple
        a list or tuple of 2 sc.AnnData object.
    
    vars_use: list or tuple
        a list or tuple of 2 variable name-lists.
        for example, differentail expressed genes, highly variable features.
    
    df_varmap: pd.DataFrame
        pd.DataFrame with 2 columns.
        relationships between features in 2 datasets, for making the 
        adjacent matrix (`vv_adj`) between variables from these 2 datasets. 
    
    df_varmap_1v1: None, pd.DataFrame; optional.
        dataframe containing only 1-to-1 correspondance between features
        in 2 datasets, if not provided, it will be inferred from `df_varmap`
    
    oo_adjs:
        a sequence of (sparse) adjacent matrices of observations.

    vars_as_nodes: list or tuple of 2
        variables to be taken as the graph nodes

    union_node_feats: bool or 'auto'
        whether to take the union of the variable-nodes

    dataset_names: list or tuple of 2
        names to discriminate data source, e.g. ('reference', 'query')

    with_single_vnodes: bool
        whether to include the varibales (node) that are ocurred in only one of
        the datasets

    Returns
    -------
    dpair: DataPair
    
    Examples
    --------
    >>> dpair = datapair_from_adatas(
    ...     [adata1, adata2],
    ...     vars_use = [hvgs1, hvgs2],
    ...     df_varmap = homo_gene_matches,
    ...     dataset_names = ['reference', 'query']
    ...     )

    """
    adata1, adata2 = adatas
    adata_raw1 = adata1.raw.to_adata() if adata1.raw is not None else adata1
    adata_raw2 = adata2.raw.to_adata() if adata2.raw is not None else adata2

    if vars_as_nodes is None:
        vars_as_nodes = vars_use
    ### features selected for modeling. (e.g. DEGs, HVGs)
    vars_use1, vars_use2 = vars_use
    vars_nodes1, vars_nodes2 = vars_as_nodes
    ### --- obs. annotation dataframes
    obs1 = adata1.obs.copy()
    obs2 = adata2.obs.copy()

    ### --- deal with variable mapping dataframe
    if df_varmap_1v1 is None:
        print('1-to-1 mapping between variables (`df_varmap_1v1`) is not '
              'provided, extracting from `df_varmap`')
        df_varmap_1v1 = utp.take_1v1_matches(df_varmap)
    ### --- connection between variables from 2 datasets
    vars_all1, vars_all2 = adata_raw1.var_names, adata_raw2.var_names
    submaps = utp.subset_matches(df_varmap, vars_nodes1, vars_nodes2, union=True)
    submaps = utp.subset_matches(submaps, vars_all1, vars_all2, union=False)

    if with_single_vnodes:
        vv_adj, vnodes1, vnodes2 = utp.make_bipartite_adj(
            submaps, vars_nodes1, vars_nodes2,
            with_singleton=True, symmetric=True,
        )
    else:
        vv_adj, vnodes1, vnodes2 = utp.make_bipartite_adj(
            submaps, with_singleton=False, symmetric=True,
        )
        ### --- ov_adjs (unaligned features, for making `ov_adj`)
    ov_adjs1 = adata_raw1[:, vnodes1].X  # for ov_adj
    ov_adjs2 = adata_raw2[:, vnodes2].X
    var1 = adata1.var.copy().loc[vnodes1, :]
    var2 = adata2.var.copy().loc[vnodes2, :]

    ### --- node features 
    if union_node_feats == 'auto' and sum(map(len, vars_use)) < 3000:
        union_node_feats = True
    else:
        union_node_feats = False
    if union_node_feats:
        submaps_1v1 = utp.subset_matches(
            df_varmap_1v1, vars_use1, vars_use2, union=True)
        # make sure that all the features are detected in both datasets
        submaps_1v1 = utp.subset_matches(
            submaps_1v1, vars_all1, vars_all2, union=False)
    else:
        # (intersection of 1-1 matched freatures)
        submaps_1v1 = utp.subset_matches(
            df_varmap_1v1, vars_use1, vars_use2, union=False)
    vnames_feat1, vnames_feat2 = list(zip(*submaps_1v1.values[:, :2]))

    #    try:
    #        features1 = adata1[:, vnames_feat1].X
    #        features2 = adata2[:, vnames_feat2].X
    #    except:
    print('[NOTE]\nthe node features will be extracted from `adata.raw`, '
          'please make sure that the values are normalized.\n')
    features1 = adata_raw1[:, vnames_feat1].X
    features2 = adata_raw2[:, vnames_feat2].X

    features = list(map(_check_sparse_toarray, [features1, features2]))

    return DataPair(features,
                    [ov_adjs1, ov_adjs2],
                    vv_adj=vv_adj,
                    oo_adjs=oo_adjs,
                    varnames_feat=[vnames_feat1, vnames_feat2],
                    varnames_node=[vnodes1, vnodes2],
                    obs_dfs=[obs1, obs2],
                    var_dfs=[var1, var2],
                    dataset_names=dataset_names,
                    **kwds)

