# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:14:05 2020

@author: Xingyan Liu
====================================================

Functions for downstream biological analysis


Example:
    import utils_analyze as uta
    uta.fooo(...)

"""

import os
from pathlib import Path
from typing import Sequence, Union, Mapping, Optional, Callable
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy import sparse
import networkx as nx

import scanpy as sc
from . import preprocess as utp
from . import plot as uplt
from ..datapair.unaligned import DataPair
from . import _knn
from .base import make_nowtime_tag, make_pairs_from_lists


# In[]


def _unique_cats(labels, cats=None, ):
    uniques = np.unique(labels)  # sorted

    if cats is None:
        if hasattr(labels, 'cat'):
            cat_ord = labels.cat.categories
        elif isinstance(labels, pd.Categorical):
            cat_ord = labels.categories
        else:
            cat_ord = uniques  # sorted
        return [c for c in cat_ord if c in uniques]

    else:
        return cats


def _check_str_seq(s, n=2):
    if isinstance(s, str):
        return [s] * n
    return s


def _int2str(lst):
    return list(map(str, lst))


def _int_to_cat_strings(lst):
    cats = _int2str(np.unique(lst))  # ordered by default
    lst = _int2str(lst)
    return pd.Categorical(lst, categories=cats)


# In[]


class ResultsContainer(object):

    def __init__(self, name='results_container'):
        self.name = name

    def keys(self, ):
        return [k for k in dir(self) if not k.startswith('_') and k != 'keys']

    def __getitem__(self, key, default=None):
        return getattr(self, key, default)

    def __setitem__(self, key, value):
        print(f'[{self.name}] Setting item to key `{key}`')
        setattr(self, key, value)

    def __str__(self, ):
        s = '\n'.join([f'{self.__class__}',
                       f'name: {self.name}',
                       f'keys: {self.keys()}'])
        return s


class ResultsAnalyzer(object):
    """
    
    ==== Attributes ====

    obs:
        * original labels (used for comparison)
        * predicted labels (with or without `rejection`)
        * predicted probabilities
        * (is_right, is_right_rej)
        * (other user provided annoations)
        other annotaions:
        * original_name: names of observations (e.g. cell barcodes)
        * dataset: which dataset each observation belongs to
        
    var:
        * name: names of variables (e.g. gene symbols)
        * dataset: which dataset each variable belongs to
        * is_linked: whether a variable has any correspondences in the other
            dataset
        * is_linked_1v1: whether a variable has any 1-v-1 correspondences in 
            the other dataset
        * (other user-provided annoations)
        
    h_obs:
        hidden states of the obseravtions from the model outputs
    h_var:
        hidden states of the variables from the model outputs
    
    
    ==== Methods ====
    
    Computational:
    * confusion matrix  (with or without row-normalization)
    * UMAP for observations
    * UMAP for variables
    * averaged expressions of each variable in different observation-groups, 
      added to `.var`
    * correlations (consine distances) of each homologous pairs
    * (top k) homologous pairs with the highest / lowest correlations
    * clustering of variables
    
    Plot:
    * confusion matrix
    * alluvial diagram
    * UMAP
    * bipartite abstracted graph (TODO !!!)
    
    I/O:
    * correlations (consine distances) of each homologous pairs
    * (top k) homologous pairs with the highest / lowest correlations
    
    """

    def __init__(
            self,
            dpair: DataPair,
            hidden_dict: Mapping[str, np.ndarray],
            obs: Union[pd.DataFrame, Mapping, None] = None,
            var: Union[pd.DataFrame, Mapping, None] = None,
            resdir: Union[str, Path, None] = None,
            name: Union[str, None] = None,
            key_dataset: str = None,
            key_obs: str = 'cell',
            key_var: str = 'gene',
            **kwds):

        self.KEY_OBS = key_obs
        self.KEY_VAR = key_var

        # setting results
        self.set_resdir(resdir)
        self.set_datapair(dpair)
        self.set_hidden_states(hidden_dict)
        if obs is not None: self.set_obs_annos(obs)
        if var is not None: self.set_var_annos(var)
        #        self.set_dataset_names(dataset_names)

        self.make_obs_hdata()
        self.make_var_hdata()

        if name is None:
            self.name = '({},{})'.format(*self.dataset_names)

        self.extra = ResultsContainer(name='extra_results-' + self.name)
        self.params = ResultsContainer(name='analyze_params-' + self.name)
        self.summary()

    def summary(self, ):
        attr_names = [
            'KEY_OBS', 'KEY_VAR', 'dataset_names',
            'resdir', 'name',
        ]
        for nm in attr_names:
            attr = getattr(self, nm)
            print(f'.{nm} = {attr}')

        print()
        print('Variables are annotated from the following views:\n\t',
              list(self.var.columns))
        print('Observations are annotated from the following views:\n\t',
              list(self.obs.columns))

    def set_datapair(self, dpair: DataPair):
        self.dpair = dpair
        self.reset_dpair()

    def reset_dpair(self, dpair: Optional[DataPair] = None):
        if dpair is None:
            dpair = self.dpair
        else:
            self.dpair = dpair
        self.vv_adj = dpair._vv_adj
        self.KEY_DATASET = dpair.KEY_DATASET
        self.KEY_OBSNAME = dpair.KEY_OBSNAME
        self.KEY_VARNAME = dpair.KEY_VARNAME
        self.dataset_names = dpair.dataset_names
        self.obs_ids1 = dpair.get_obs_ids(0, astensor=False)
        self.obs_ids2 = dpair.get_obs_ids(1, astensor=False)
        self.var_ids1 = dpair.get_vnode_ids(0, astensor=False)
        self.var_ids2 = dpair.get_vnode_ids(1, astensor=False)
        self.set_obs_annos(dpair.obs)
        self.set_var_annos(dpair.var)

        self.var_names1 = self.var.loc[self.var_ids1, self.KEY_VARNAME]
        self.var_names2 = self.var.loc[self.var_ids2, self.KEY_VARNAME]

    def set_resdir(self, resdir=None):
        if resdir is None:
            tag = make_nowtime_tag()
            resdir = Path(f'results-{self.name}-{tag}')
        else:
            resdir = Path(resdir)
        self.resdir = resdir
        self.figdir = resdir  # / 'figs' # TODO!
        utp.check_dirs(self.figdir)
        sc.settings.figdir = self.figdir
        print('setting default directory for results')

    def set_hidden_states(self, hidden_dict: Mapping[str, np.ndarray], ):
        """ Setting hidden states for observations (cells) and variables (genes)
        """
        self.h_obs = hidden_dict[self.KEY_OBS]
        self.h_var = hidden_dict[self.KEY_VAR]

    def set_obs_annos(self, obs: pd.DataFrame,
                      clear_old=False,
                      only_diff=True,
                      ignore_index=True,
                      **kwannos):
        if not hasattr(self, 'obs') or clear_old:
            # setting for the first time
            self.obs = obs.copy()
            self._set_annos(self.obs, None,
                            ignore_index=ignore_index,
                            copy=False, **kwannos)
        else:
            self._set_annos(self.obs, obs,
                            only_diff=only_diff,
                            ignore_index=ignore_index,
                            copy=False, **kwannos)

    def set_var_annos(self, var: pd.DataFrame,
                      clear_old=False,
                      only_diff=True,
                      ignore_index=True,
                      **kwannos):

        if not hasattr(self, 'var') or clear_old:
            self.var = var.copy()
            self._set_annos(self.var, None,
                            ignore_index=ignore_index,
                            copy=False, **kwannos)
        else:
            self._set_annos(self.var, var,
                            only_diff=only_diff,
                            ignore_index=ignore_index,
                            copy=False, **kwannos)

    def get_obs_anno(self, key, which=None):

        if which is not None:
            obs_ids = self.obs_ids1 if which == 0 else self.obs_ids2
        if key in self.obs.columns:
            values = self.obs.loc[obs_ids, key]
        else:
            values = self.dpair.get_obs_anno(key, which=which, concat=True)

        return values

    def get_var_anno(self, key, which=None):

        if which is not None:
            var_ids = self.var_ids1 if which == 0 else self.var_ids2
        #        if key in self.obs.columns:
        values = self.var.loc[var_ids, key]
        return values

    def make_obs_hdata(self, ):
        self.hdata_obs = utp.make_adata(self.h_obs, obs=self.obs.copy(),
                                        assparse=False)

    def make_var_hdata(self, ):
        self.hdata_var = utp.make_adata(self.h_var, obs=self.var.copy(),
                                        assparse=False)
        self.hdata_var_all = self.hdata_var

    def split_obs(self, by=None, left_groups=None, ):
        if by is None: by = self.KEY_DATASET
        if left_groups is None: left_groups = self.dataset_names[0]

        adt1, adt2 = utp.bisplit_adata(self.hdata_obs, by,
                                       left_groups=left_groups)
        return adt1, adt2

    def split_obs_by_dataset(self, set_attr=True):
        by = self.KEY_DATASET
        left_groups = self.dataset_names[0]

        adt1, adt2 = utp.bisplit_adata(self.hdata_obs, by,
                                       left_groups=left_groups)
        adt1.obs_names = adt1.obs[self.KEY_OBSNAME]
        adt2.obs_names = adt2.obs[self.KEY_OBSNAME]
        if set_attr:
            self.hdata_obs1, self.hdata_obs2 = adt1, adt2
            print('results are set as attributes: `.hdata_obs1` and `.hdata_obs2`')
        else:
            return adt1, adt2

    def split_var_by_dataset(self, set_attr=True):
        by = self.KEY_DATASET
        left_groups = self.dataset_names[0]
        adt1, adt2 = utp.bisplit_adata(self.hdata_var, by,
                                       left_groups=left_groups)
        # when splitted by `dataset`, no worries about var-name collision
        adt1.obs_names = self.var_names1
        adt2.obs_names = self.var_names2
        for adt in (adt1, adt2):
            adt.obs['is_private'] = pd.Categorical(
                adt.obs['is_linked'].apply(lambda x: not x),
                categories=[False, True])

        if set_attr:
            self.hdata_var1, self.hdata_var2 = adt1, adt2
            print('results are set as attributes: `.hdata_var1` and `.hdata_var2`')
        else:
            return adt1, adt2

    #    def

    # In[]
    """     functions for computation
    ========================================================
    """

    def confusion_matrix(self, key1, key2, which=1,
                         classes_on=None,
                         normalize=None, **kwds):
        """ confusion matrix (with or without row-normalization)
        """
        lbs_y = self.get_obs_anno(key1, which=which)  # y_true
        lbs_x = self.get_obs_anno(key2, which=which)

        mat = wrapper_confus_mat(
            lbs_y, lbs_x, classes_on=classes_on,
            normalize=normalize,
            **kwds)

        return mat

    def contingency_mat(self, key1, key2, which=1,
                        order_cols=True,
                        order_rows=False,
                        normalize_axis=None,
                        **kwds):
        lbs_y = self.get_obs_anno(key1, which=which)  # y_true
        lbs_x = self.get_obs_anno(key2, which=which)

        mat = wrapper_contingency_mat(
            lbs_y, lbs_x,
            order_cols=order_cols,
            order_rows=order_rows,
            normalize_axis=normalize_axis,
            as_df=True,
            **kwds)

        return mat

    def compute_obs_umap(self, metric='cosine',
                         n_neighbors=15,
                         params_umap={},
                         adjust=False,
                         return_data=False,
                         **kwds):
        """Computing UMAP embeddings from the hidden-states of the observations,
        the results will be backed up into:
        (if adjust)
            .extra.obs_distances_adjusted
            .extra.obs_connectivities_adjusted
            .extra.obs_umap_adjusted
        (if not adjust)
            .extra.obs_distances
            .extra.obs_connectivities
            .extra.obs_umap
            
        
        **kwds: other papameters for `paired_data_neighbors`;
            ks=10, ks_inner=5, binarize = False, 
            func_norm = None, algorithm = 'auto', ...
        """
        adt = self.hdata_obs  # .copy()
        #        key_dist0 = 'distances'
        #        key_conn0 = 'connectivities'
        if adjust:  # TODO: random seeds
            _key = 'adjusted'
            _ = paired_data_neighbors(
                self.h_obs[self.obs_ids1], self.h_obs[self.obs_ids2],
                adata=adt,
                metric=metric,
                key_added=_key,
                **kwds)
            sc.tl.umap(adt, neighbors_key=_key, **params_umap)
            dist, conn = get_adata_neighbors(adt, key=_key)

        else:
            _key = None
            sc.pp.neighbors(adt, metric=metric, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.umap(adt, neighbors_key=_key, **params_umap)
            dist, conn = get_adata_neighbors(adt, key=_key)

        tag = '' if _key is None else '_' + _key
        self.extra['obs_distances' + tag] = dist
        self.extra['obs_connect' + tag] = conn
        self.extra['obs_umap' + tag] = adt.obsm['X_umap']

        if return_data: return adt

    def subset_vars(self, var_ids):
        #        self.hdata_var_all = self.hdata_var
        self.var_ids1 = np.array([x for x in self.var_ids1 if x in var_ids])
        self.var_ids2 = np.array([x for x in self.var_ids2 if x in var_ids])
        self.hdata_var = self.hdata_var[var_ids, :].copy()
        self.vv_adj = self.vv_adj[var_ids, :][:, var_ids]
        self.var = self.var.iloc[var_ids, :]

        self.var_names1 = self.var.loc[self.var_ids1, self.KEY_VARNAME]
        self.var_names2 = self.var.loc[self.var_ids2, self.KEY_VARNAME]

    def compute_var_neighbors(self,
                              n_neighbors=8, metric='cosine',
                              use_rep='X',
                              exact=True,
                              adjust=2,
                              key_added=None):
        """
        the results will be backed up into:
            .extra.var_distances
            .extra.var_connect
            .extra.var_connect_adjusted 
        
        """
        #        key_added = 'integrative' if adjust else None
        adt = self.hdata_var
        self.split_var_by_dataset()
        adt1, adt2 = self.hdata_var1, self.hdata_var2

        for _adt in [adt1, adt2]:
            adata_neighbors(_adt, metric=metric,
                            n_neighbors=n_neighbors,
                            exact=exact,
                            use_rep=use_rep,
                            key_added=None)
        dist1, conn1 = get_adata_neighbors(adt1, key=None)
        dist2, conn2 = get_adata_neighbors(adt2, key=None)
        # combine adjacent matrix
        dist = sparse.block_diag([dist1, dist2]).tocsr()
        conn = sparse.block_diag([conn1, conn2]).tocsr()
        print(type(conn))

        #        if adjust:
        homonet = utp.normalize_row(self.vv_adj, by='sum')
        if isinstance(adjust, (int, float)):
            homonet *= adjust
        net = conn + homonet  # strengthen the homo-links
        #            net = conn.maximum(homonet)
        set_precomputed_neighbors(
            adt, dist, net,
            n_neighbors=n_neighbors,
            metric=metric,
            use_rep=use_rep,
            key_added=key_added,
        )

        # record
        self.extra.var_connect_adjusted = net

        self.extra.var_distances = dist
        self.extra.var_connect = conn

        self.params['compute_var_umap'] = dict(
            n_neighbors=n_neighbors,
            adjust=adjust,
            neighbors_key=key_added,
        )

    def compute_var_umap(
            self,
            force_redo=False,
            metric='cosine',
            n_neighbors=5,
            adjust=2,
            return_data=False,
            #           key_added='integrative',
            **kwds):
        """ Computing UMAP embeddings from the hidden-states of the variables,
        the results will be backed up into:
            .extra.var_distances
            .extra.var_connect
            .extra.var_connect_adjusted (if adjust is True)
            .extra.var_umap
        """
        adt = self.hdata_var
        key_added = None
        if force_redo or 'connectivities' not in adt.obsp.keys():
            self.compute_var_neighbors(
                n_neighbors=n_neighbors, metric=metric,
                adjust=adjust, )
        #
        #        use_rep='X'
        #        key_added = 'integrative' if adjust else None
        ##        key_dist0 = f'{key_added}_distances'
        ##        key_conn0 = f'{key_added}_connectivities'
        #        sc.pp.neighbors(adt, metric = metric,
        #                        n_neighbors=n_neighbors,
        #                        use_rep=use_rep,
        #                        key_added=None)
        #
        #        dist, conn = get_adata_neighbors(adt, key=None)
        #
        #        if adjust:
        #            homonet = utp.normalize_row(self.vv_adj, by='sum')
        #            if isinstance(adjust, (int, float)):
        #                homonet *= adjust
        #            net = conn + homonet # strengthen the homo-links
        ##            net = conn.maximum(homonet)
        #            set_precomputed_neighbors(
        #                    adt, dist, net,
        #                    n_neighbors=n_neighbors,
        #                    metric=metric,
        #                    use_rep=use_rep,
        #                    key_added=key_added,
        #                    )
        #            self.extra.var_connect_adjusted = net

        sc.tl.umap(adt, neighbors_key=key_added, **kwds)
        self.hdata_var1.obsm['X_umap'] = adt.obsm['X_umap'][self.var_ids1, :]
        self.hdata_var2.obsm['X_umap'] = adt.obsm['X_umap'][self.var_ids2, :]

        self.extra.var_umap = adt.obsm['X_umap']

        #        print(adt)
        if return_data: return adt

    def weight_linked_vars(self, metric='cosine',  # only1v1=False,
                           func_dist2weight=None,
                           sigma=None,
                           return_df=True,
                           sort=True,
                           **kwds):
        """ correlations (consine distances) of each linked (homologous) pairs
        of variables.
        
        returns
        -------
        pd.DataFrame with columns:
            (name1, name2), corr, is1v1
        """
        #        from scipy.spatial.distance import cdist

        vv_adj = sparse.triu(self.vv_adj).tocoo()
        X1 = self.h_var[vv_adj.row, :]
        X2 = self.h_var[vv_adj.col, :]

        foo = lambda x1, x2: cdist(x1[None, :], x2[None, :], metric=metric)[0, 0]
        dists = np.array(list(map(foo, X1, X2)))

        # transfrom dists to weights
        if func_dist2weight is None:
            if (dists > 1).any():  # for 'euclidean' distances
                sigma = 2 * np.var(dists) if sigma is None else sigma
                #                print(sigma)
                weights = np.exp(-np.square(dists) / sigma)
            else:
                weights = 1 - dists
        else:
            weights = func_dist2weight(dists)

        # constructing the resulting dataframe
        names_all = self.var['name']
        index = pd.MultiIndex.from_tuples(
            list(zip(np.take(names_all, vv_adj.row),
                     np.take(names_all, vv_adj.col))),
            names=self.dataset_names)
        df = pd.DataFrame({'distance': dists, 'weight': weights, },
                          index=index)
        # is 1-v-1 homologous or not
        names1 = index.get_level_values(0)
        names_freq1 = utp.take_freq1(names1)
        df['is1v1'] = [nm in names_freq1 for nm in names1]
        if sort:
            print('sorting links by weights')
            df.sort_values(by='weight', ascending=False, inplace=True)

        self.var_link_weights = df
        print('the resulting distances and weights between linked-pairs '
              'are stored in `.var_link_weights`')

        # record parameters
        params = dict(
            metric=metric,
            func_dist2weight=func_dist2weight,
            sigma=sigma,
        )
        self.params['weight_linked_vars'] = params

        return df if return_df else None

    #    def cluster_vars(self, resolution=0.8,
    def _jointly_extract_var_modules(
            self, resolution=0.8,
            method='leiden',
            key_added='joint_module',
            neighbors_key=None,  # or 'integrative'
            **kwds):
        """ jointly cluster vars
        """
        #        if neighbors_key is None:
        #            neighbors_key = self.params['compute_var_umap']['neighbors_key']
        cluster_func = {'leiden': sc.tl.leiden,
                        'louvain': sc.tl.louvain}[method]
        cluster_func(self.hdata_var, resolution=resolution,
                     key_added=key_added,
                     neighbors_key=neighbors_key,
                     **kwds)

        ### record results and parameters
        self.set_var_annos(var=self.hdata_var.obs)
        if not hasattr(self, 'hdata_var1'):
            self.split_var_by_dataset(set_attr=True)
        else:
            self.hdata_var1.obs[key_added] = self.hdata_var.obs[key_added][self.var_ids1].values
            self.hdata_var2.obs[key_added] = self.hdata_var.obs[key_added][self.var_ids2].values

        params = dict(
            resolution=resolution,
            method=method,
            key_added=key_added,
        )
        if len(kwds) >= 1:
            params.update(kwds)
        self.params['cluster_vars'] = params

        self.extra['var_cluster_counts'] = utp.group_value_counts(
            self.hdata_var.obs, key_added, self.KEY_DATASET)
        print(self.extra.var_cluster_counts)

    def _separately_extract_var_modules(
            self, nneigh=5,
            resolution=0.8,
            keys=['sep_module', 'degree'],
            method='leiden',
            resplit=True,
            **kwds):
        """
        for each dataset, build a KNN-graph for variables, and detect modules
        using 'leiden' or 'louvain' algorithm.
        
        output: 
            A pair of dataframes indexed by var-names
        """
        if not hasattr(self, 'hdata_var1'):
            self.split_var_by_dataset(set_attr=True)

        mods = []
        for adt in (self.hdata_var1, self.hdata_var2):
            mod, inter_mod_conn = _extract_modules(
                adt, nneigh=nneigh, use_rep='X',
                resolution=resolution, key_added=keys[0],
                method=method, copy=False, **kwds)
            mods.append(mod)
        #        mod1, mod2 = mods
        self.params['_extract_var_modules'] = dict(
            nneigh=nneigh,
            resolution=resolution,
            keys=keys,
            method=method,
            resplit=resplit,
        )

        return mods

    def extract_modules(self, nneigh=5,
                        resolution=0.8, resplit=True,
                        keys=['sep_module', 'degree'],
                        method='leiden',
                        **kwds):

        mod1, mod2 = self._separately_extract_var_modules(
            nneigh=nneigh, resolution=resolution, resplit=resplit,
            keys=keys, method=method, **kwds)
        key_mod, key_degree = keys

        if not hasattr(self, 'var_link_weights'):
            self.weight_linked_vars()
        df_links = self.var_link_weights
        mod_lbs1, mod_lbs2, mod_conn = _match_groups(
            mod1[key_mod], mod2[key_mod], df_links, key_weight='weight')

        self.hdata_var1.obs[key_mod] = mod_lbs1
        self.hdata_var2.obs[key_mod] = mod_lbs2

        mod_annos = pd.concat(
            [self.hdata_var1.obs[keys], self.hdata_var2.obs[keys]], axis=0)
        print(mod_annos)
        for k in keys:
            self.hdata_var.obs[k] = mod_annos[k].values
        self.set_var_annos(var=self.hdata_var.obs)

    def extract_consensus_modules(
            self,
            resolution=0.8,
            cut_score=0.5,
            cut_member=0.5,  # 0.4 if you'd like to keep more genes
            keys=['joint_module', 'sep_module'],
            key_added='module',
            key_member='membership',
    ):
        if not isinstance(resolution, Sequence):
            resolution = [resolution] * 2
        self._jointly_extract_var_modules(resolution=resolution[0])
        #        self._separately_extract_var_modules()
        self.extract_modules(resolution=resolution[1])
        null = 'filtered'  # np.nan
        mod_labels = []
        mod_member = []
        for adt in (self.hdata_var1, self.hdata_var2):
            _annos = adt.obs
            _annos[key_added] = _annos[keys].apply(lambda x: '_'.join(x), axis=1)

            scores, memberships = module_scores(adt, key_added, min_nodes=3)
            _annos[key_member] = memberships

            # filter out low-density modules
            labels_flt = _annos[key_added].apply(
                lambda x: x if scores[x] > cut_score else null)

            # filter points with low memberships
            labels_flt[_annos[key_member] <= cut_member] = null
            _annos[key_added] = labels_flt
            mod_labels.append(labels_flt)
            mod_member.append(_annos[key_member].values)

        mod_lbs1, mod_lbs2 = mod_labels  # np.array
        kept1, kept2 = (mod_lbs1 != null, mod_lbs2 != null)
        #        kept1, kept2 = (mod_lbs1.notna(), mod_lbs2.notna())
        print('kept1.sum(), kept2.sum()', kept1.sum(), kept2.sum())

        # match and re-order
        if not hasattr(self, 'var_link_weights'):
            self.weight_linked_vars()
        df_links = self.var_link_weights
        #        print('------matching modules-------')
        mod_lbs1[kept1], mod_lbs2[kept2], mod_conn = _match_groups(
            mod_lbs1[kept1], mod_lbs2[kept2], df_links,
            key_weight='weight', as_cats=False)

        mod_lbs1[~kept1] = null
        mod_lbs2[~kept2] = null
        # record
        self.hdata_var1.obs[key_added] = mod_lbs1
        self.hdata_var2.obs[key_added] = mod_lbs2
        mod_lbs = np.hstack([mod_lbs1.values, mod_lbs2.values])
        self.hdata_var.obs[key_added] = mod_lbs
        self.hdata_var.obs[key_member] = np.hstack(mod_member)
        self.set_var_annos(var=self.hdata_var.obs)
        self.extra['homo_mod_conn'] = mod_conn

    def export_modules(self, mods, ):
        key = self.params['_extract_var_modules']['keys'][0]

        pass

    def compute_averages(self, groupby: Union[str, Sequence[str]],
                         mat=None, ):
        """
        mat: if provided, should be a matrix of shape (self.n_obs, self.n_var)
        """
        # obs_labels
        groupby_obs1, groupby_obs2 = _check_str_seq(groupby, 2)
        if not hasattr(self, 'hdata_obs1'):
            self.split_obs_by_dataset(set_attr=True)
        obs_labels1 = self.hdata_obs1.obs[groupby_obs1]
        obs_labels2 = self.hdata_obs2.obs[groupby_obs2]

        ### exprssion matrix or attention matrix
        if mat is None:
            mat = self.dpair._ov_adj.tocsr()  # exprssion matrix
        mat1 = mat[self.obs_ids1, :][:, self.var_ids1]
        mat2 = mat[self.obs_ids2, :][:, self.var_ids2]

        avgs1 = utp.group_mean(mat1, obs_labels1, features=self.var_names1, )
        avgs2 = utp.group_mean(mat2, obs_labels2, features=self.var_names2, )

        return avgs1, avgs2

    def make_abstracted_graph(
            self,
            groupby_obs: Union[str, Sequence],  # = 'celltype'
            groupby_var: Union[None, str, Sequence] = None,
            avg_exprs=None,  # if provided, should be a tuple of 2 pd.DataFrame
            obs_group_order1=None, obs_group_order2=None,
            var_group_order1=None, var_group_order2=None,
            tags_obs: Union[None, Sequence[str]] = None,
            tags_var: Union[None, Sequence[str]] = None,
            key_weight='weight',
            key_count='size',
            key_identity='identity',
            cut_ov=0.5,  # (0.55, 2.5)
            norm_mtd_ov='zs',  # 'zs' or 'max'
            **kwds):

        # obs_labels
        groupby_obs1, groupby_obs2 = _check_str_seq(groupby_obs, 2)
        if not hasattr(self, 'hdata_obs1'):
            self.split_obs_by_dataset(set_attr=True)

        obs_labels1 = self.hdata_obs1.obs[groupby_obs1]
        obs_labels2 = self.hdata_obs2.obs[groupby_obs2]

        # var_labels
        if groupby_var is None:
            groupby_var = 'module'
        groupby_var1, groupby_var2 = _check_str_seq(groupby_var, 2)

        if not hasattr(self, 'hdata_var1'):
            self.split_var_by_dataset(set_attr=True)

        var_labels1 = self.hdata_var1.obs[groupby_var1]
        var_labels2 = self.hdata_var2.obs[groupby_var2]

        if tags_obs is None:
            tags_obs = (f'{s} ' for s in self.dataset_names)
        if tags_var is None:
            tags_var = (f'{s} module ' for s in self.dataset_names)
        self.params['tags_obs'] = tags_obs
        self.params['tags_var'] = tags_var

        # average expressions
        if avg_exprs is None:
            print('computing averages for pairs of observation-varible using '
                  'default settings')
            avg_expr1, avg_expr2 = self.compute_averages(
                groupby=(groupby_obs1, groupby_obs2))
        else:
            avg_expr1, avg_expr2 = avg_exprs

        if not hasattr(self, 'var_link_weights'):
            print('computing weights for links of variable pairs using '
                  'default settings')
            self.weight_linked_vars()
        df_var_links = self.var_link_weights.copy()

        g = make_abstracted_graph(
            obs_labels1, obs_labels2,
            var_labels1, var_labels2,
            avg_expr1, avg_expr2,
            df_var_links,
            #                obs_group_order1 = obs_group_order1,
            #                obs_group_order2 = obs_group_order2,
            var_group_order1=var_group_order1,
            var_group_order2=var_group_order2,
            tags_obs=tags_obs,
            tags_var=tags_var,
            key_weight=key_weight,
            key_count=key_count,
            key_identity=key_identity,
            cut_ov=cut_ov,
            norm_mtd_ov=norm_mtd_ov,
        )
        self.abstracted_graph = g
        return g

    def get_module_connection(self, ):  # TODO: subset graph --> to_pandas_edgelis
        import networkx as nx
        #        from networkx import to_pandas_edgelist
        g = self.abstracted_graph
        tag1, tag2 = self.params['tags_var']
        edge_df = nx.to_pandas_edgelist(g)
        kept = edge_df[['source', 'target']].apply(
            lambda x: tag1 in x[0] and tag2 in x[1], axis=1)
        return edge_df.loc[kept, :]

    def write_graph(self, fname=None, suffix='.gpickle', resdir=None):
        """ saving the abstracted graph
        """
        from networkx.readwrite.gpickle import write_gpickle
        g = self.abstracted_graph

        if '.gpickle' == suffix:
            write_func = write_gpickle
        if not str(fname).endswith(suffix):
            fname += suffix
        write_func(g, fname)  # TODO: loading graph

    # In[] ================== visualization functions ================
    def plot_confusion_mat(
            self, key1, key2, which=1,
            classes_on=None,
            normalize='true',
            linewidths=0.01, linecolor='grey',
            fn_mat=None,
            fp=None,
            **kwds):
        mat = self.confusion_matrix(key1, key2, which=which,
                                    classes_on=classes_on,
                                    normalize=normalize)
        if fn_mat is not None:
            fp_mat = self.resdir / fn_mat
            mat.to_csv(fp_mat)
            print(f'Confusion matrix was saved into {fp_mat}')

        ax = uplt.heatmap(mat, linewidths=linewidths, linecolor=linecolor,
                          xlabel=key2, ylabel=key1,  # note the order
                          square=True, fp=fp, **kwds)
        return ax, mat

    def plot_contingency_mat(
            self, key1, key2, which=1,
            normalize_axis=1,
            linewidths=0.01, linecolor='grey',
            fn_mat=None,
            fp=None,
            order_cols=True, order_rows=True,
            **kwds):
        """
        A Contingency matrix does not assume the same classes
        """
        mat = self.contingency_mat(key1, key2, which=which,
                                   order_cols=order_cols, order_rows=order_rows,
                                   normalize_axis=normalize_axis,
                                   )
        if fn_mat is not None:
            fp_mat = self.resdir / fn_mat
            mat.to_csv(fp_mat)
            print(f'Contingency matrix was saved into {fp_mat}')

        ax = uplt.heatmap(mat, linewidths=linewidths, linecolor=linecolor,
                          xlabel=key2, ylabel=key1,  # note the order
                          square=True, fp=fp, **kwds)
        return ax, mat

    def plot_alluvial(
            self, key1, key2, which=1,
            classes_on=None,
            labels=None,
            title=None,
            normalize_axis=None,
            cmap='tab20_r',
            fn_mat=None,
            savefig=True,
            fn_fig=None,
            **kwds):
        """ Using Contingency Matrix, other than confusion matrix!!!
        """
        mat = self.contingency_mat(key1, key2, which=which,
                                   order_cols=False,
                                   normalize_axis=normalize_axis,
                                   )
        if fn_mat is not None:
            fp_mat = self.resdir / fn_mat
            mat.to_csv(fp_mat)
            print(f'Contingency matrix was saved into {fp_mat}')

        if title is None:
            tt = 'predicted classes of {} \n({} as the reference)'.format(
                *self.dataset_names[::-1])

        if savefig:
            if fn_fig is None:
                fn_fig = f'alluvial-{which}-{(key1, key2)}.png'
            else:
                fn_fig = fn_fig
            fp = self.figdir / fn_fig
        else:
            fp = None

        if labels is None: labels = (key1, key2)
        ax = uplt.alluvial_plot(mat, title=tt, cmap_name=cmap, labels=labels,
                                fname=fp, **kwds)
        return ax, mat

    def plot_obs_umap(self, color, adjust=False,
                      save=None, **kwds):

        adt = self.hdata_obs
        tag = '_adjusted' if adjust else ''
        X_umap = self.extra[f'obs_umap{tag}']
        set_adata_obsm(adt, X_umap, 'X_umap')

        params = dict(legend_fontsize=11, ncols=1, save=save)
        if len(kwds) >= 1:
            params.update(kwds)
        return sc.pl.umap(adt, color=color, **params)

    def plot_var_umap(self, color,  # adjust=True,
                      save=None, **kwds):

        return sc.pl.umap(self.hdata_var, color=color, save=save, **kwds)

    def plot_var_triple_umaps(self, tags=None, fname=None, **kwds):

        if tags is None:
            tag1, tag2 = self.dataset_names
        else:
            tag1, tag2 = tags

        if not hasattr(self, 'hdata_var1'):
            self.split_var_by_dataset()
        adt1, adt2 = self.hdata_var1, self.hdata_var2

        if isinstance(fname, str):
            #            fname = 'gene_embeddings_with_private_hlighted.png'
            fp = self.figdir / fname

        colors = ['dataset', 'is_private', 'is_private']
        fig = uplt.triple_umaps(
            adt1, adt2, colors=colors,
            titles=[f'merged {tag1} and {tag2} genes',
                    f'(the private are colored)\n{tag1} genes',
                    f'{tag2} genes', ],
            fp=fp, **kwds)
        return fig

    def plot_obstracted_graph(self, **kwds):  # TODO !!!
        """bipartite abstracted graph 
        """
        pass

    # ================== I/O functions ================
    def export_all(self, ):

        self.export_obs_annos()
        self.export_var_annos()
        self.export_var_link_weights()

    def export_var_annos(self, fname=None, resdir=None, tag=None):
        """ Annotations of ALL the variables from both datasets.

        If `fname` is not given, it will be decided autometically, and `tag`
        can be set to avoid file over-written.
        """
        attr_name = 'var'
        self.set_var_annos(var=self.hdata_var.obs)
        df = getattr(self, attr_name)
        self._export_df(df, fname, resdir=resdir,
                        prefix=attr_name,
                        suffix='.csv',
                        tag=tag,
                        header=True, index=True,
                        index_label='node_id')

    def export_obs_annos(self, fname=None, resdir=None, tag=None):
        """ Annotations of ALL the observations from both datasets.

        If `fname` is not given, it will be decided autometically, and `tag`
        can be set to avoid file over-written.
        """
        attr_name = 'obs'
        df = getattr(self, attr_name)
        self._export_df(df, fname, resdir=resdir,
                        prefix=attr_name,
                        suffix='.csv',
                        tag=tag,
                        header=True, index=True,
                        index_label='node_id')

    def export_var_link_weights(self, fname=None, resdir=None, tag=None):
        """ correlations (consine distances) of each homologous pairs.

        If `fname` is not given, it will be decided autometically, and `tag`
        can be set to avoid file over-written.
        """
        attr_name = 'var_link_weights'
        df = getattr(self, attr_name)
        self._export_df(df, fname, resdir=resdir,
                        prefix=attr_name,
                        suffix='.csv',
                        tag=tag,
                        header=True, index=True,
                        index_label=self.dataset_names)

    def export_abstracted_edges(self, fname=None, resdir=None, tag=None):  # TODO: change
        """
        
        If `fname` is not given, it will be decided autometically, and `tag`
        can be set to avoid file over-written.
        """
        import networkx as nx
        g = self.abstracted_graph
        df = nx.to_pandas_edgelist(g)
        self._export_df(df, fname, resdir=resdir,
                        prefix='abstracted_graph_edges',
                        suffix='.csv',
                        tag=tag,
                        header=True, index=False,
                        index_label=self.dataset_names)

    def export_module_connection(self, fname=None, resdir=None, tag=None):
        """
        
        If `fname` is not given, it will be decided autometically, and `tag`
        can be set to avoid file over-written.
        """
        #        import networkx as nx
        #        g = self.abstracted_graph
        df = self.get_module_connection()
        self._export_df(df, fname, resdir=resdir,
                        prefix='module_connection',
                        suffix='.csv',
                        tag=tag,
                        header=True, index=False,
                        index_label=self.dataset_names)

    # ==============================================
    def _export_df(self, df: pd.DataFrame,
                   fname: Union[str, Path, None] = None,
                   resdir: Union[str, Path, None] = None,
                   prefix='', suffix='.csv',  # ignored if fname is not None
                   tag=None,
                   **kwds):
        if fname is None:
            if tag is None:
                fname = f'{prefix}{suffix}'
            else:
                fname = f'{prefix}-{tag}{suffix}'

        resdir = self.resdir if resdir is None else Path(resdir)
        fp = resdir / fname
        df.to_csv(fp, **kwds)
        print(f'dataframe have been saved into: {fp}')

    def save(self, fn='results.pickle', fp=None):

        if fp is None: fp = self.resdir / fn
        import pickle
        with open(fp, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def _set_annos(df0, df=None,
                   only_diff=True,
                   ignore_index=True,
                   copy=False, **kwannos):

        def __set_values(_df, k, v, ignore_index=ignore_index):
            if k in _df.columns:
                print(f'NOTE that column "{k}" will be covered by new values')
            _df[k] = list(v) if ignore_index else v

        if copy: df0 = df0.copy()
        if df is not None:
            if only_diff:
                cols = [c for c in df.columns if c not in df0.columns]
            else:
                cols = df.columns
            for col in cols:
                __set_values(df0, col, df[col])

        if len(kwannos) >= 1:
            for k, v in kwannos.items():
                # be careful of the index mismatching problems!!!
                __set_values(df0, k, v)
        return df0 if copy else None


# In[]
""" link-weights between homologous gene pairs """


def weight_linked_vars(
        X: np.ndarray,
        adj: sparse.spmatrix,
        names: Optional[Sequence] = None,
        metric='cosine',  # only1v1=False,
        func_dist2weight: Optional[Callable] = None,
        sigma: Optional[float] = None,
        return_df=True,
        sort=True,
        index_names=(0, 1),
        **kwds):
    """ correlations (consine distances) of each linked (homologous) pairs
    of variables.
    X: 
        np.ndarray;
        feature matrix of shape (N, M), where N is the number of sample and M is
        the feature dimensionality.
    adj: 
        sparse.spmatrix; binary adjacent matrix of shape (N, N).
    names: 
        a sequence of names;
        names for rows of `X`, of shape (N,)
    
    returns
    -------
    pd.DataFrame with columns:
        (name1, name2), distance, weight
    """
    adj = sparse.triu(adj).tocoo()

    foo = lambda x1, x2: cdist(x1[None, :], x2[None, :], metric=metric)[0, 0]
    dists = np.array(list(map(foo, X[adj.row, :], X[adj.col, :])))

    # transfrom dists to weights
    if func_dist2weight is None:
        if metric == 'euclidean':  # for 'euclidean' distances
            # print('TEST')
            sigma = 2 * np.var(dists) if sigma is None else sigma
            #                print(sigma)
            weights = np.exp(-np.square(dists) / sigma)
        else:
            weights = 1 - dists
    else:
        weights = func_dist2weight(dists)

    # constructing the resulting dataframe
    if names is None:
        index = pd.MultiIndex.from_tuples(
            list(zip(adj.row, adj.col)), names=index_names)
    else:
        index = pd.MultiIndex.from_tuples(
            list(zip(np.take(names, adj.row), np.take(names, adj.col))),
            names=index_names)
    df = pd.DataFrame({'distance': dists, 'weight': weights, },
                      index=index)
    if sort:
        print('sorting links by weights')
        df.sort_values(by='weight', ascending=False, inplace=True)

    return df if return_df else None


# In[]
""" compute module eigen-vector
"""


def _compute_svd(X, k=1, only_comps=True, whiten=False, **kwds):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=k, whiten=whiten, **kwds)
    pca.fit_transform(X)

    comps = pca.components_
    if k == 1:
        comps = comps[0]

    if only_comps:
        return comps
    else:
        return pca


def _compute_svd_eigen_corr(X, whiten=False, **kwds):
    """
    compute the first eigen-vector and the correlations between each "sample"
    (row of X) and the eigen-vector.
    """
    comps = _compute_svd(X, k=1, only_comps=True, whiten=whiten, **kwds)
    corr = 1 - cdist(X, comps[None, :], metric='correlation')[:, 0]
    #    corr = 1 - cdist(X, comps[None,:], metric='cosine')[:, 0]

    return comps, corr


def compute_group_eigens(X, labels, groups=None, whiten=False, **kwds):
    """ compute eigen-vector for each group
    groups:
        if provided, compute only for these groups, with the order kept.
    """
    groups = np.unique(labels, ) if groups is None else groups

    eigens = {}
    corrs = []
    for lb in groups:
        ind = labels == lb
        #        print(ind)
        if sum(ind) == 1:
            print(f'skipping class {lb} with only one sample.')
            v = X[ind, :].flatten()
            corrs.append(np.array([1]))
            eigens[lb] = v
            continue
        print(lb)
        v, corr = _compute_svd_eigen_corr(X[ind, :], whiten=whiten, **kwds)
        eigens[lb] = v
        corrs.append(corr.flatten())

    eigen_df = pd.DataFrame(eigens)  # each column as a eigen-vector
    memberships = np.hstack(corrs)

    return eigen_df, memberships


# In[]
# """     KNN-searching for each dataset separately, and conbined with homo-links
# """
# def independ_neighbors(
#        adata,
#        split_by='dataset',
#        n_neighbors=8, metric='cosine',
#        use_rep='X',
#        exact=True,
#        adjust=2,
#        key_added = None):

def jointly_extract_modules(
        adata,
        resolution=0.8,
        method='leiden',
        key_added='cluster',
        neighbors_key=None,  # or 'integrative'
        **kwds):
    pass


# In[]
"""     module extraction     
==================================
"""


def _filter_for_abstract(
        var_labels1, var_labels2,
        avg_expr1, avg_expr2,
        df_var_links,
        name=None):
    if name is None:
        kept1, kept2 = tuple(map(pd.notna, (var_labels1, var_labels2)))
    else:
        kept1, kept2 = tuple(map(
            lambda x: x != name, (var_labels1, var_labels2)))

    var_labels1, var_labels2 = var_labels1[kept1], var_labels2[kept2]
    avg_expr1, avg_expr2 = avg_expr1[kept1], avg_expr2[kept2]
    vars1, vars2 = avg_expr1.index, avg_expr2.index
    kept_index = list(filter(
        lambda x: x[0] in vars1 and x[1] in vars2,
        df_var_links.index.values
    ))
    df_var_links = df_var_links.loc[kept_index, :]
    print(kept1.sum(), kept2.sum())
    return var_labels1, var_labels2, avg_expr1, avg_expr2, df_var_links,


def module_scores(adata, groupby='module',
                  neighbor_key=None,
                  min_nodes=3, ):
    _, conn = get_adata_neighbors(adata, neighbor_key)
    labels = adata.obs[groupby]
    degrees = (conn > 0).sum(1).A.flatten()
    #    avg_degree = degrees.mean()
    nets = dict()
    density = dict()
    memberships = np.zeros_like(degrees, dtype=float)  # .astype(float)
    for lb in pd.unique(labels):
        #        print(lb.center(80, '-'))
        inds = np.flatnonzero(labels == lb)
        subnet = conn[inds, :][:, inds]
        nets[lb] = subnet
        _degrees = degrees[inds]
        memberships[inds] = subnet.sum(1).A.flatten() / _degrees  # [inds]

        if len(inds) <= min_nodes:
            density[lb] = -1
        else:
            density[lb] = subnet.nnz / _degrees.sum()

    return density, memberships


def _net_density(mat: sparse.spmatrix):
    n, m = mat.shape
    if n == m:
        max_nnz = n * (n - 1)
    else:
        max_nnz = n * m
    if sparse.issparse(mat):
        nnz = mat.nnz
    else:
        nnz = (mat != 0).sum()
    return nnz / max_nnz


def _extract_modules(
        adata, nneigh=10,
        metric='cosine', use_rep='X',
        resolution=0.6, key_added='module',
        method='leiden', copy=False,
        force_redo=False,
        **kwds):
    """
    build a KNN-graph, and detect modules using 'leiden' or 'louvain' algorithm.
    * `use_rep='X'` indicates the `adata.X` should be the reducted embeddings.
    * method should either be 'leiden' or 'louvain'
    
    output: 
        mod: a pd.DataFrame indexed by `adata.obs_names` with columns: 
            ['module', 'degree']
        mod_conn: inter-module-connection
    """
    adata = adata.copy() if copy else adata
    cluster_func = {'leiden': sc.tl.leiden,
                    'louvain': sc.tl.louvain}[method]

    if 'connectivities' not in adata.obsp.keys() or force_redo:
        sc.pp.neighbors(adata, metric=metric,
                        n_neighbors=nneigh, use_rep=use_rep)

    cluster_func(adata, resolution=resolution,
                 key_added=key_added,
                 **kwds)

    # compute 
    dist, conn0 = get_adata_neighbors(adata, )
    conn = utp._binarize_mat(conn0)

    # inter-module-connection
    lbs = adata.obs[key_added]
    inter_mod_conn = utp.agg_group_edges(
        conn0, lbs, lbs, groups1=None, groups2=None, )

    # the degrees for each node
    adata.obs['degree'] = conn.sum(1).A1.flatten()

    mod = adata.obs[[key_added, 'degree']].copy()

    # re-order the modules based on the inter-module connections
    ordered_mod = order_by_similarities(inter_mod_conn)
    _map = dict(list(zip(ordered_mod, np.arange(len(ordered_mod)))))
    mod[key_added] = [_map[x] for x in list(mod[key_added])]

    adata.uns['inter_mod_conn'] = inter_mod_conn
    return mod, inter_mod_conn


def _match_groups(mod_lbs1: Union[pd.Series, Mapping],
                  mod_lbs2: Union[pd.Series, Mapping],
                  df_homo: Optional[pd.DataFrame] = None,
                  key_weight=None,
                  keys_link: Union[None, Sequence[str]] = None,
                  fix_left=True,
                  as_cats=True,
                  **kwds):
    """
    match two different sets of groups (modules), and re-arrange the group_labels
    """
    tokens1 = list(mod_lbs1.keys())
    tokens2 = list(mod_lbs2.keys())

    #    print(mod_lbs1.value_counts())
    if df_homo is None:
        unique_tokens = list(set(tokens1) + set(tokens2))
        df_homo = pd.DataFrame({
            'src': unique_tokens,
            'tgt': unique_tokens,
            'weight': 1.,
        })
        keys_link = ('src', 'tgt')
    # cross-set module connections
    mod_conn = aggregate_links(
        df_homo, mod_lbs1, mod_lbs2,
        key_weight=key_weight,
        keys_link=keys_link,
        norm_sizes='auto',
        **kwds
    )
    n1, n2 = mod_conn.shape
    # re-arrange the module-numbers
    mod_conn = utp.order_contingency_mat(mod_conn, axis=1)  # re-order the rows
    mod_conn = utp.order_contingency_mat(mod_conn, axis=0)  # re-order the columns
    #    print(mod_conn)

    # change the labels
    new_cats1 = _int2str(np.arange(n1))
    map1 = dict(list(zip(mod_conn.index, new_cats1)))
    mod_conn.index = new_cats1
    new_mod_lbs1 = np.array([map1[_lb] for _lb in mod_lbs1])

    new_cats2 = _int2str(np.arange(n2))
    map2 = dict(list(zip(mod_conn.columns, new_cats2)))
    mod_conn.columns = new_cats2
    new_mod_lbs2 = np.array([map2[_lb] for _lb in mod_lbs2])

    if as_cats:
        new_mod_lbs1 = pd.Categorical(new_mod_lbs1, categories=new_cats1)
        new_mod_lbs2 = pd.Categorical(new_mod_lbs2, categories=new_cats2)

    return new_mod_lbs1, new_mod_lbs2, mod_conn


def order_by_similarities(sims: pd.DataFrame):
    """
    sims: pd.DataFrame with the same index and columns
    """

    ordered = sims.index.tolist()
    return ordered


def _subgroup_edges(
        adj, labels, groups, node_names=None,
        key_row='source',
        key_col='target',
        key_data='weight',
        as_attrs=False):
    kept = utp.take_group_labels(labels, groups, indicate=True)
    ids = np.flatnonzero(kept)
    adj = adj[ids, :][:, ids]
    if node_names is None:
        names = ids
    else:
        names = node_names[ids]
    # transformed into edge-df
    df_edge = adj_to_edges(
        adj,
        rownames=names,
        colnames=names,
        key_row=key_row,
        key_col=key_col,
        key_data=key_data,
        as_attrs=as_attrs)

    return df_edge


def adata_subgroup_edges(
        adata, groups, groupby='module', key=None,
        **kwds):
    """
    take a subset of the KNN-graph from adata, melt to an edge-dataframe, with
    columns ['source', 'target', 'weight'] by default.
    """

    dist, conn = get_adata_neighbors(adata, key=key, )
    lbs = adata.obs[groupby]
    ndnames = adata.obs_names

    df_edge = _subgroup_edges(conn, lbs, groups, ndnames, **kwds)
    return df_edge


def edges_from_adata(
        adata, kind='conn',
        key_neigh=None,
        key_row='source',
        key_col='target',
        as_attrs=False):
    """ transform the connectivities in `adata` into an edge-dataframe, with
    columns ['source', 'target', 'weight'] by default.
    
    kind: 'conn', 'dist'
    """
    dist, conn = get_adata_neighbors(adata, key=key_neigh)
    names = adata.obs_names
    if kind == 'conn':
        df_edge = adj_to_edges(
            sparse.triu(conn),
            rownames=names,
            colnames=names,
            key_row=key_row,
            key_col=key_col,
            key_data='weight',
            as_attrs=as_attrs)
    elif kind == 'dist':
        df_edge = adj_to_edges(
            sparse.triu(dist),
            rownames=names,
            colnames=names,
            key_row=key_row,
            key_col=key_col,
            key_data='distance',
            as_attrs=as_attrs)

    return df_edge


def nx_from_adata(
        adata, key_neigh=None,
        keys_attr=None,
) -> nx.Graph:
    """ nx.Graph from the KNN graph of `adata`
    """
    node_data = adata.obs if keys_attr is None else adata.obs[keys_attr]
    edges = edges_from_adata(adata, 'conn', as_attrs=True)
    nodes = make_nx_input_from_df(node_data)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    print(nx.info(g))
    return g


def neighbor_induced_edgedf(
        df, nodes,
        keys_edge=['source', 'target'],
        return_nodes=False,
        fp=None):
    _inds = df[keys_edge].isin(nodes).max(1)
    subnodes = np.unique(df[_inds][['source', 'target']].values.flatten())
    df_sub = utp.subset_matches(df, subnodes, subnodes, cols=keys_edge, union=False)
    #    df_sub.shape
    if fp is not None:
        df_sub.to_csv(fp, index=False)

    if return_nodes:
        return df_sub, subnodes
    else:
        return df_sub


def neighbor_induced_subgraph(g: nx.Graph, nodes):
    """ Note that the returned graph also contains the edges connecting the 
    neighbor-nodes if there is any.
    """
    nodes = [n for n in nodes if n in list(g)]
    subnodes = nx_neighbors(g, nodes, with_self=True)

    return g.subgraph(subnodes)


def nx_neighbors(g, nodes, with_self=False):
    """ find all the neighbors of given nodes
    """
    if isinstance(nodes, str):
        nodes = [nodes]
    tmp = []
    for nd in nodes:
        tmp += nx.neighbors(g, nd)
    if with_self:
        tmp += list(nodes)

    return np.unique(tmp)


def nx_to_dfs(
        g: nx.Graph,
        nodelist: Union[Sequence, None] = None,
        source="source", target="target",
        **kwds):
    """
    outputs:
        edgedf, nodedf
    example:
    >>> edgedf, nodedf = nx_to_dfs(g)
    """
    if isinstance(nodelist, str):
        nodelist = [nodelist]

    edgedf = nx.to_pandas_edgelist(
        g, source=source, target=target, nodelist=nodelist)
    if nodelist is not None:
        nodelist = np.unique(edgedf[[source, target]].values.flatten())
    nodedf = nx_to_nodedf(g, nodelist)
    return edgedf, nodedf


def nx_to_nodedf(g, nodelist=None):
    nodedf = pd.DataFrame(g.nodes.values(), index=g.nodes.keys())
    if nodelist is not None:
        nodedf = nodedf.loc[nodelist, :]
    return nodedf


def export_subgraph_df(
        g, nodenames=None, fdir=Path('.'), tag='', tfset=None):
    """
    export graph to '.csv' files for other visualization
    """
    if nodenames is not None:
        g = nx.subgraph(g, nodenames)
    print(nx.info(g))
    edgedf, nodedf = nx_to_dfs(g, )
    if tfset is not None:
        nodedf['TF'] = nodedf['name'].isin(tfset)
    fpe = fdir / f'nxEdges-{tag}.csv'
    fpn = fdir / f'nxAnnos-{tag}.csv'
    edgedf.to_csv(fpe, index=False)
    nodedf.to_csv(fpn, index=False)
    print(fpe, fpn, sep='\n')
    return g


def export_neighbor_subgraph_df(
        g, nodenames=None, fdir=Path('.'), tag='', tfset=None):
    """
    temp function; 
    """
    if nodenames is not None:
        g = neighbor_induced_subgraph(g, nodenames)
    print(nx.info(g))
    edgedf, nodedf = nx_to_dfs(g, )
    if tfset is not None:
        nodedf['TF'] = nodedf['name'].isin(tfset)
    fpe = fdir / f'gene_edges-{tag}.csv'
    fpn = fdir / f'gene_annos-{tag}.csv'
    edgedf.to_csv(fpe, index=False)
    nodedf.to_csv(fpn, index=False)
    print(fpe, fpn, sep='\n')
    return g


# In[]
"""     functions for cell/gene embeddings
==================================================
"""


def set_adata_obsm(adata, X_h, key_add='X_h', copy=False):
    """
    key_add: which key will be added to adata.obsm
        probably be 'X_umap', 'X_h', 'X_pca', etc.
        
    Example:
    >>> set_adata_obsm(adata, h_cell, 'X_h')
    >>> sc.pp.neighbors(adata, metric='cosine', n_neighbors=15, use_rep='X_h')
    >>> sc.tl.umap(adata)
    >>> sc.pl.umap(adata, color='cell_type')
    """
    if copy:
        print('Making a copy.')
        adata = adata.copy()
    else:
        print('No copy was made.')

    adata.obsm[key_add] = X_h
    return adata if copy else None


def get_adata_neighbors(adata, key: Union[str, None] = None):
    """
    getting distances and connectivities from adata
    """
    key_dist = 'distances'
    key_conn = 'connectivities'
    if key is not None:
        key_dist = f'{key}_' + key_dist
        key_conn = f'{key}_' + key_conn

    return adata.obsp[key_dist], adata.obsp[key_conn]


# In[]
"""     abstracted graph
===================================
"""


def write_graph_cyjs(g, fp='tmp.ctjs', return_dct=False, attrs=None, **kwds):
    """ Cytoscape Json format
    """
    from networkx.readwrite.json_graph import cytoscape_data

    dct = cytoscape_data(g, attrs)
    utp.save_json_dict(dct, fp, **kwds)
    print(fp)
    return dct if return_dct else None


def make_abstracted_graph(
        obs_labels1, obs_labels2,
        var_labels1, var_labels2,
        avg_expr1, avg_expr2,
        df_var_links,
        #        obs_group_order1 = None,# TODO ? deciding orders
        #        obs_group_order2 = None,
        var_group_order1=None,
        var_group_order2=None,
        tags_obs=('', ''),
        tags_var=('', ''),
        key_weight='weight',
        key_count='size',
        key_identity='identity',
        cut_ov=0.55,
        norm_mtd_ov='max',
        global_adjust_ov=True,
        global_adjust_vv=True,
        vargroup_filtered='filtered',
        **kwds):
    """
    obs_labels1, obs_labels2,
    
    var_labels1, var_labels2,
    
    avg_expr1, avg_expr2,
    
    df_var_links,
    
    norm_mtd_ov: one of {None, 'zs', 'maxmin', 'max'}
    """
    tag_obs1, tag_obs2 = tags_obs
    tag_var1, tag_var2 = tags_var
    var_labels1, var_labels2, avg_expr1, avg_expr2, df_var_links = \
        _filter_for_abstract(
            var_labels1, var_labels2, avg_expr1, avg_expr2, df_var_links,
            name=vargroup_filtered)
    #    obs_group_order1 = _unique_cats(obs_labels1, obs_group_order1)
    #    obs_group_order2 = _unique_cats(obs_labels2, obs_group_order2)
    var_group_order1 = _unique_cats(var_labels1, var_group_order1)
    var_group_order2 = _unique_cats(var_labels2, var_group_order2)
    print('--->', var_group_order1)
    # obs-var edge abstraction #
    edges_ov1, avg_vo1 = abstract_ov_edges(
        avg_expr1, var_labels1,
        norm_method=norm_mtd_ov,
        cut=cut_ov,
        tag_var=tag_var1, tag_obs=tag_obs1,
        global_adjust=global_adjust_ov,
        return_full_adj=True)
    edges_ov2, avg_vo2 = abstract_ov_edges(
        avg_expr2, var_labels2,
        norm_method=norm_mtd_ov,
        cut=cut_ov,
        tag_var=tag_var2, tag_obs=tag_obs2,
        global_adjust=global_adjust_ov,
        return_full_adj=True)
    print('---> avg_vo1\n', avg_vo1)
    # var-weights abstraction #
    edges_vv, adj_vv = abstract_vv_edges(
        df_var_links,  # res.var_link_weights,
        var_labels1,
        var_labels2,
        norm_sizes='auto',
        #            norm_sizes=(df_vnodes1[key_count], df_vnodes2[key_count]),
        return_full_adj=True,
        global_adjust=global_adjust_vv,
        key_weight=key_weight,
        tag_var1=tag_var1,
        tag_var2=tag_var2,
        **kwds)
    print('---> adj_vv\n', adj_vv)

    # deciding orders of the groups #
    avg_vo1 = utp.order_contingency_mat(avg_vo1, 1)  # 1 for the rows (vars)
    avg_vo1 = utp.order_contingency_mat(avg_vo1, 0)
    var_group_order1, obs_group_order1 = avg_vo1.index, avg_vo1.columns

    var_name_order1 = [f'{tag_var1}{x}' for x in var_group_order1]
    print(var_name_order1)
    adj_vv = utp.order_contingency_mat(adj_vv.loc[var_name_order1, :], 0)
    var_group_order2 = [x.replace(tag_var2, '') for x in adj_vv.columns]
    if set(obs_group_order1) == set(obs_labels2):
        obs_group_order2 = obs_group_order1
    else:
        avg_vo2 = utp.order_contingency_mat(avg_vo2.loc[var_group_order2, :], 0)
        obs_group_order2 = avg_vo2.columns

    # obs-nodes abstraction #    
    node_attrs_obs1 = abstract_nodes(
        obs_labels1, obs_group_order1, tag=tag_obs1)
    node_attrs_obs2 = abstract_nodes(
        obs_labels2, obs_group_order2, tag=tag_obs2)

    # var-nodes abstraction #    
    node_attrs_var1, df_vnodes1 = abstract_nodes(
        var_labels1, var_group_order1, tag=tag_var1,
        return_df=True)
    node_attrs_var2, df_vnodes2 = abstract_nodes(
        var_labels2, var_group_order2, tag=tag_var2,
        return_df=True)

    df_vnodes1, df_vnodes2 = list(map(lambda x: x.set_index(key_identity),
                                      [df_vnodes1, df_vnodes2]))

    # var-weights abstraction #
    edges_vv, adj_vv = abstract_vv_edges(
        df_var_links,  # res.var_link_weights,
        var_labels1,
        var_labels2,
        #            norm_sizes=(df_vnodes1[key_count], df_vnodes2[key_count]),
        return_full_adj=True,
        global_adjust=global_adjust_vv,
        key_weight=key_weight,
        tag_var1=tag_var1,
        tag_var2=tag_var2,
        **kwds)

    # graph construction #
    g = nx_multipartite_graph(
        node_attrs_obs1,
        node_attrs_var1, node_attrs_var2,
        node_attrs_obs2,
        edges=edges_ov1 + edges_vv + edges_ov2)

    return g


def abstract_vv_edges(
        df_links: pd.DataFrame,
        var_labels1: Union[pd.Series, Mapping],
        var_labels2: Union[pd.Series, Mapping],
        norm_sizes: Union[None, Sequence, str] = 'auto',
        return_full_adj=True,
        global_adjust: bool = True,
        global_norm_func=max,
        key_weight: str = 'weight',
        keys_link: Union[None, Sequence[str]] = None,
        tag_var1: str = '',
        tag_var2: str = '',
):
    """
    df_links: pd.DataFrame, shape=(n_edges, *)
        If `keys_edge` is provided, it should be a tuple of 2 column-names
        in df_links.columns, indicating the edge columns. 
        Otherwise, the `df_links.index` should be `pd.MultiIndex` to indicate 
        the source and target edges.
        This can also be output form ResultsAnalyzer.weight_linked_vars(), 
        or the stored attribute `ResultsAnalyzer.var_link_weights`
    
    keys_edge: If `keys_edge` is provided, it should be a tuple of 2 column-names
        in df_links.columns, indicating the edge columns.
    var_labels1, var_labels2:
        grouping labels for the two sets of variables, respectively.
    """
    #    _var_labels1 = pd.Series({k: f'{tag_var1}{c}' for k, c in var_labels1.items()})
    #    _var_labels2 = pd.Series({k: f'{tag_var1}{c}' for k, c in var_labels2.items()})

    abs_vv = aggregate_links(
        df_links, var_labels1, var_labels2, norm_sizes=None,  # normalize latter
        key_weight=key_weight, keys_link=keys_link)

    abs_vv.index = [f'{tag_var1}{c}' for c in abs_vv.index]
    abs_vv.columns = [f'{tag_var2}{c}' for c in abs_vv.columns]

    if norm_sizes is not None:
        if norm_sizes == 'auto':
            _var_labels1 = [f'{tag_var1}{c}' for c in var_labels1]
            _var_labels2 = [f'{tag_var2}{c}' for c in var_labels2]
            sizes1, sizes2 = [pd.value_counts(_lbs) for _lbs in [_var_labels1, _var_labels2]]
        #            print(sizes1, sizes2)
        elif isinstance(norm_sizes, Sequence):
            sizes1, sizes2 = norm_sizes
        #        print(abs_vv)
        abs_vv = weight_normalize_by_size(
            abs_vv, sizes1, sizes2, global_adjust=global_adjust,
            norm_func=global_norm_func)

    edges_vv = adj_to_edges(abs_vv, as_attrs=True)
    if return_full_adj:
        return edges_vv, abs_vv
    else:
        return edges_vv


def aggregate_links(
        df_links: pd.DataFrame,
        labels1: Union[pd.Series, Mapping],
        labels2: Union[pd.Series, Mapping],
        norm_sizes: Union[None, Sequence, str] = None,
        global_adjust: bool = False,
        global_norm_func=max,
        key_weight: str = 'weight',
        keys_link: Union[None, Sequence[str]] = None,
):
    """
    df_links: pd.DataFrame, shape=(n_edges, *)
        If `keys_edge` is provided, it should be a tuple of 2 column-names
        in df_links.columns, indicating the edge columns. 
        Otherwise, the `df_links.index` should be `pd.MultiIndex` to indicate 
        the source and target edges.
        This can also be output form ResultsAnalyzer.weight_linked_vars(), 
        or the stored attribute `ResultsAnalyzer.var_link_weights`
    
    keys_edge: If `keys_edge` is provided, it should be a tuple of 2 column-names
        in df_links.columns, indicating the edge columns.
        
    labels1, labels2:
        grouping labels for the rows and columns, respectively.
        
    norm_sizes:
        if provided, should be a pair of pd.Series, dict, Mapping, list
        it can be also set as 'auto', which means decide sizes by the group-labels.
    """
    rnames0 = labels1.keys()
    cnames0 = labels2.keys()

    if keys_link is not None:
        df_links = df_links.set_index(keys_link)

    #    print(key_weight, df_links, sep='\n')
    _data = df_links[key_weight] if key_weight in df_links.columns else None
    #    print(_data)
    adj_var, rnames, cnames = utp.pivot_to_sparse(
        rows=df_links.index.get_level_values(0),
        cols=df_links.index.get_level_values(1),
        data=_data,
        rownames=rnames0, colnames=cnames0)
    # make sure the labels are correctly ordered
    lbs1 = np.array([labels1[r] for r in rnames])  # var_labels1[rnames]
    lbs2 = np.array([labels2[c] for c in cnames])
    #    print(pd.value_counts(lbs1, dropna=False))
    aggregated = utp.agg_group_edges(
        adj_var, lbs1, lbs2, groups1=None, groups2=None, )

    if norm_sizes is None:
        return aggregated

    if norm_sizes == 'auto':
        sizes1, sizes2 = [pd.value_counts(_lbs) for _lbs in [lbs1, lbs2]]
    elif isinstance(norm_sizes, Sequence):
        sizes1, sizes2 = norm_sizes
    aggregated = weight_normalize_by_size(
        aggregated, sizes1, sizes2,
        global_adjust=global_adjust, norm_func=global_norm_func)

    return aggregated


def weight_normalize_by_size(adj, sizes1, sizes2,
                             norm_func=max,  # denominator
                             global_adjust=False):
    """
    adj: pd.DataFrame, shape = (n1, n2)
    sizes1, sizes2: pd.Series, dict, Mapping, list-like
        shape = (n1,) and (n2,)
    global_adjust: bool; whether to perform a global adjustment after 
        size-normalization.
    """
    if not isinstance(adj, pd.DataFrame): adj = pd.DataFrame(adj)
    if isinstance(sizes1, list):
        sizes1 = dict(list(zip(adj.index, sizes1)))
    if isinstance(sizes2, list):
        sizes2 = dict(list(zip(adj.columns, sizes2)))

    _adj = adj.copy()
    for r, c in make_pairs_from_lists(_adj.index, _adj.columns, skip_equal=False):
        _adj.loc[r, c] /= norm_func([sizes1[r], sizes2[c]])

    if global_adjust:
        _adj /= _adj.max().max()
        # abs_vv = utp.wrapper_normalize(abs_vv, 'max', axis=1)

    return _adj


def abstract_ov_edges(
        avg_expr,
        var_labels,
        norm_method=None,
        norm_axis=1,
        groupby_var='__temp_labels__',  # 'module',
        tag_var='',
        tag_obs='',
        cut=0,
        global_adjust: bool = False,
        return_full_adj=False):
    """
    avg_expr: pd.DataFrame; each column represent the average expressoions 
        for each observation group, and each row as a variable.
    norm_method: one of {None, 'zs', 'maxmin', 'max'}
    """
    df = avg_expr.copy()
    if norm_method is not None:
        df = utp.wrapper_normalize(df, method=norm_method, axis=norm_axis)

    # averaged by varible-groups
    df[groupby_var] = var_labels
    df_avg = df.groupby(groupby_var).mean()
    df_avg.dropna(inplace=True)
    df_avg0 = df_avg.copy()

    df_avg.index = [f'{tag_var}{i}' for i in df_avg.index]
    df_avg.columns = [f'{tag_obs}{i}' for i in df_avg.columns]

    if cut is not None:
        if isinstance(cut, Sequence):
            df_avg[df_avg <= cut[0]] = 0
            df_avg[df_avg > cut[1]] = cut[1]
            print(f'Edges with weights out of range {cut} were cut out '
                  f'or clipped')
        else:
            df_avg[df_avg <= cut] = 0

    if global_adjust:
        df_avg /= df_avg.max().max()

    edge_attrs = adj_to_edges(df_avg, cut=0, as_attrs=True)
    if return_full_adj:
        return edge_attrs, df_avg0
    else:
        return edge_attrs


def adj_to_edges(
        adj,
        rownames=None,
        colnames=None,
        key_row='source',
        key_col='target',
        key_data='weight',
        cut=None,
        as_attrs=False):
    """
    Melt (Un-pivot) a pd.DataFrame of adjacent matrix into a list of edges, works
    similar to `pd.melt()` but ignoring ALL of the zeros instead.
    """
    if isinstance(adj, pd.DataFrame):
        rownames = adj.index
        colnames = adj.columns
        data = adj.values
    else:
        rownames = np.arange(adj.shape[0]) if rownames is None else rownames
        colnames = np.arange(adj.shape[1]) if colnames is None else colnames
        data = adj

    coo_data = sparse.coo_matrix(data)
    df_edge = pd.DataFrame({
        key_row: np.take(rownames, coo_data.row),
        key_col: np.take(colnames, coo_data.col),
        key_data: coo_data.data,
    })
    if cut is not None:
        if isinstance(cut, Sequence):
            df_edge = df_edge[df_edge[key_data] > cut[0]]
            df_edge[df_edge[key_data] > cut[1]] = cut[1]
            print(f'Edges with weights out of range {cut} were cut out '
                  f'or clipped')
        else:
            df_edge = df_edge[df_edge[key_data] > cut]
            print(f'Edges with weights lower than {cut} were cut out.')
    if as_attrs:
        edge_attrs = make_nx_input_from_df(df_edge, key_id=[key_row, key_col])
        return edge_attrs
    else:
        return df_edge


def abstract_nodes(labels,  # df, groupby,
                   group_ord=None,
                   key_count='size',
                   key_identity='identity',
                   key_orig='original name',
                   key_tag='tag',
                   tag='',
                   return_df=False,
                   **attrs):
    """
    Taking each of the groups in `labels` as an abstracted node, and making a 
    list of node attributes for further input to `networkx.Graph.add_nodes_from()`
    
    ==== inputs ====
    key_identity: key name for the unique identifier for each abstracted node.
    key_orig: key name for the original (group) name for each abstracted node.
    **kwds: ignored currently
    
    === Example ====
    >>> df_nodes = abstract_nodes(df, groupby='module')
    >>> g = nx.Graph()
    >>> g.add_nodes_from(node_attrs, 'node name')

    """
    #    labels = df[groupby]
    if group_ord is None:
        df_nodes = labels.value_counts().to_frame(key_count)
    else:
        df_nodes = labels.value_counts()[group_ord].to_frame(key_count)
    df_nodes[key_identity] = [f'{tag}{i}' for i in df_nodes.index]
    df_nodes[key_orig] = list(df_nodes.index)
    #    if tag != '':
    df_nodes[key_tag] = tag.strip()

    node_attrs = make_nx_input_from_df(df_nodes, key_id=key_identity, **attrs)
    if return_df:
        return node_attrs, df_nodes
    else:
        return node_attrs


def make_nx_input_from_df(
        df: pd.DataFrame,
        key_id: Union[None, str, list] = None,
        **attrs):
    """
    
    df: pd.DataFrame with each column containing node attributes
        
    key_id: 
        which column(s) should be used for unique identifier for each nodes.
        str or list of strings from df.columns, or None (using `df.index` in this case)
        If specified, the values should be unique.
    **attrs: 
        other common attributes for this batch of nodes
    
    
    === return ===
    A list of tuples, with each tuple formed like `(node, attrdict)` when 
    `len(key_id) == 1` or `(u, v, attrdict)` when `len(key_id) == 2` 
        
        
    === Example ====
    >>> node_attrs = make_nx_node_input(df, **attrs)
    >>> g = nx.Graph()
    >>> g.add_nodes_from(node_attrs)
    
    """
    if key_id is not None:
        df = df.set_index(key_id, drop=True)

    else:
        df = df.copy()
    if len(attrs) >= 1:
        for k, v in attrs.items():
            df[k] = v

    dct = df.apply(lambda x: x.to_dict(), axis=1).to_dict()
    if isinstance(df.index, pd.MultiIndex):
        return [k + (vd,) for k, vd in dct.items()]
    else:
        return [(k, vd) for k, vd in dct.items()]


def nx_multipartite_graph(*node_layers,
                          edges=None,
                          subset_key='subset', **attrs):
    """
    
    
    === Example ====
    >>> g = uta.nx_multipartite_graph([0, 1], [2, 3, 4, 5], [6, 7, 8], )
    >>> pos = nx.multipartite_layout(g, subset_key=subset_key, )
    >>> nx.draw(g, pos, with_labels=True, )
    """
    import networkx as nx

    #    print(nodes)
    g = nx.Graph()
    for i, nodes in enumerate(node_layers):
        g.add_nodes_from(nodes, **{subset_key: i})
    if edges is None:
        print('`edges` are not provides, adding full connected edges '
              'between adjacent layers by default')
        from itertools import product
        for layer1, layer2 in nx.utils.pairwise(node_layers):
            g.add_edges_from(product(layer1, layer2))
    else:
        g.add_edges_from(edges)
    return g


# In[]


def wrapper_confus_mat(y_true, y_pred, classes_on=None,
                       normalize='true', as_df=True):
    """ 
    normalize: 'true', 'pred', 'all', None
        by default, normalized by row (true classes)
    """
    from sklearn import metrics
    if classes_on is None:
        classes_on = np.unique(list(y_true) + list(y_pred))
    #        classes_on = list(set(y_true).union(y_pred))
    #    mat = metrics.confusion_matrix(y_true, y_pred, labels = classes_on,
    #                               normalize=normalize)
    try:
        mat = metrics.confusion_matrix(y_true, y_pred, labels=classes_on,
                                       normalize=normalize)
    except:
        print('probably the argument `normalize` was not accepted by the previous '
              'version of scikit-learn')
        mat = metrics.confusion_matrix(y_true, y_pred, labels=classes_on, )
    if as_df:
        mat = pd.DataFrame(mat, index=classes_on, columns=classes_on)

    return mat


def wrapper_contingency_mat(y_true, y_pred,
                            order_rows=True,
                            order_cols=False,
                            normalize_axis=None,
                            as_df=True,
                            eps=None, assparse=False):
    """ 
    Modified and wrapped function from `sklearn`:
    >>> mat = sklearn.metrics.cluster.contingency_matrix(
            y_true, y_pred, eps=eps, sparse=assparse)
    """
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(y_true, return_inverse=True)
    clusters, cluster_idx = np.unique(y_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    mat = sparse.coo_matrix((np.ones(class_idx.shape[0]),
                             (class_idx, cluster_idx)),
                            shape=(n_classes, n_clusters),
                            dtype=np.int)
    if assparse:
        mat = mat.tocsr()
        mat.sum_duplicates()
        if normalize_axis is not None:  # 0 for columns
            mat = utp.normalize_norms(mat, axis=normalize_axis)
    else:
        mat = mat.toarray()
        if eps is not None:
            # don't use += as mat is integer
            mat = mat + eps
        if normalize_axis is not None:  # 0 for columns
            mat = utp.normalize_norms(mat, axis=normalize_axis)

        if as_df:
            mat = pd.DataFrame(mat, index=classes, columns=clusters)
        # reorder so that make clusters and classes matching each other as possible
        if order_cols:
            mat = utp.order_contingency_mat(mat, 0)
        if order_rows:
            mat = utp.order_contingency_mat(mat, 1)
    return mat


# In[]
"""     KNN searching with batch considered
=====================================================
"""


def adata_neighbors(adata,
                    n_neighbors=8,
                    metric='cosine',
                    exact=False,
                    #                    algorithm = None, #'brute',
                    use_rep='X',
                    key_added=None,
                    **kwds):
    if use_rep == 'X':
        X = adata.X
    else:  # 'X_pca'
        X = adata.obsm[use_rep]

    if exact:
        dist_mat, conn = _knn.find_neighbors(
            X, n_neighbors=n_neighbors,
            metric=metric, algorithm='brute', **kwds)
        set_precomputed_neighbors(
            adata, dist_mat, conn, n_neighbors=n_neighbors,
            metric=metric, use_rep=use_rep, key_added=key_added)
    else:
        sc.pp.neighbors(adata, metric=metric,
                        n_neighbors=n_neighbors, use_rep=use_rep,
                        key_added=key_added)

    return adata  # although chaneged inplace


def paired_data_neighbors(
        X1, X2,
        adata: Union[sc.AnnData, None] = None,
        ks=10, ks_inner=3,
        binarize=False,
        n_pcs: Optional[int] = None,
        use_rep: Optional[str] = None,
        random_state=0,
        method='umap',
        metric='cosine',
        metric_kwds=None,
        key_added: Union[None, str] = 'adjusted',
        **kwds) -> sc.AnnData:
    """
    X1, X2: np.ndarray, shape = (n_obs1, n_var) and (n_obs2, n_var)
    adata: if provided, should contain (n_obs1 + n_obs2) data observations, and
        each columns should be matched with `X1` followed by `X2`!!!
    
    """
    if adata is None:
        X = np.vstack([X1, X2])
        adata = sc.AnnData(X=X, )
    #    if use_rep == 'X_pca':
    #        sc.tl.pca(adata, n_comps=n_pcs)

    distances, connectivities = _knn.pair_stitched_knn(
        X1, X2,
        ks=ks,
        ks_inner=ks_inner,
        metric=metric,
        #        func_norm = 'umap',
        algorithm='auto',
        metric_params=metric_kwds,
        **kwds)
    if binarize:
        connectivities[connectivities > 0] = 1

    n_neighbors = ks[0] if isinstance(ks, Sequence) else ks
    set_precomputed_neighbors(
        adata,
        distances,
        connectivities,
        n_neighbors=n_neighbors,
        metric=metric,
        method=method,  # 'umap',
        metric_kwds=metric_kwds,
        use_rep=use_rep,
        n_pcs=n_pcs,
        key_added=key_added,
    )

    return adata  # Always return data


def set_precomputed_neighbors(
        adata,
        distances,
        connectivities=None,
        n_neighbors=15,
        metric='cisone',  # pretended parameter
        method='umap',  # pretended parameter
        metric_kwds=None,  # pretended parameter
        use_rep=None,  # pretended parameter
        n_pcs=None,  # pretended parameter
        key_added=None,  #
):
    if key_added is None:
        key_added = 'neighbors'
        conns_key = 'connectivities'
        dists_key = 'distances'
    else:
        conns_key = key_added + '_connectivities'
        dists_key = key_added + '_distances'

    if connectivities is None:
        connectivities = distances.copy().tocsr()
        connectivities[connectivities > 0] = 1

    adata.obsp[dists_key] = distances
    adata.obsp[conns_key] = connectivities

    adata.uns[key_added] = {}
    neighbors_dict = adata.uns[key_added]
    neighbors_dict['connectivities_key'] = conns_key
    neighbors_dict['distances_key'] = dists_key

    neighbors_dict['params'] = {'n_neighbors': n_neighbors, 'method': method}
    neighbors_dict['params']['metric'] = metric
    if metric_kwds is not None:
        neighbors_dict['params']['metric_kwds'] = metric_kwds
    if use_rep is not None:
        neighbors_dict['params']['use_rep'] = use_rep
    if n_pcs is not None:
        neighbors_dict['params']['n_pcs'] = n_pcs

    return adata
