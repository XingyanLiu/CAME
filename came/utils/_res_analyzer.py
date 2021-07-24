# -*- coding: UTF-8 -*-
"""
@author: Xingyan Liu
@file: _res_analyzer.py
@time: 2021-06-14
"""
from . import preprocess as pp
from . import plot as uplt
from .analyze import *
from .analyze import (
    _match_groups,
    _extract_modules,
)


def _check_str_seq(s, n=2):
    if isinstance(s, str):
        return [s] * n
    return s


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
        pp.check_dirs(self.figdir)
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

        if key in self.obs.columns:
            if which is not None:
                obs_ids = self.obs_ids1 if which == 0 else self.obs_ids2
                values = self.obs.loc[obs_ids, key]
            else:
                values = self.obs[key]
        else:
            values = self.dpair.get_obs_anno(key, which=which, concat=True)

        return values

    def get_var_anno(self, key, which=None):

        if which is not None:
            var_ids = self.var_ids1 if which == 0 else self.var_ids2
            values = self.var.loc[var_ids, key]
        else:
            values = self.var[key]
        return values

    def make_obs_hdata(self, ):
        self.hdata_obs = pp.make_adata(self.h_obs, obs=self.obs.copy(),
                                        assparse=False)

    def make_var_hdata(self, ):
        self.hdata_var = pp.make_adata(self.h_var, obs=self.var.copy(),
                                        assparse=False)
        self.hdata_var_all = self.hdata_var

    def split_obs(self, by=None, left_groups=None, ):
        if by is None: by = self.KEY_DATASET
        if left_groups is None: left_groups = self.dataset_names[0]

        adt1, adt2 = pp.bisplit_adata(self.hdata_obs, by,
                                       left_groups=left_groups)
        return adt1, adt2

    def split_obs_by_dataset(self, set_attr=True):
        by = self.KEY_DATASET
        left_groups = self.dataset_names[0]

        adt1, adt2 = pp.bisplit_adata(self.hdata_obs, by,
                                       left_groups=left_groups)
        adt1.obs_names = adt1.obs[self.KEY_OBSNAME]
        adt2.obs_names = adt2.obs[self.KEY_OBSNAME]
        if set_attr:
            self.hdata_obs1, self.hdata_obs2 = adt1, adt2
            print(
                'results are set as attributes: `.hdata_obs1` and `.hdata_obs2`')
        else:
            return adt1, adt2

    def split_var_by_dataset(self, set_attr=True):
        by = self.KEY_DATASET
        left_groups = self.dataset_names[0]
        adt1, adt2 = pp.bisplit_adata(self.hdata_var, by,
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
            print(
                'results are set as attributes: `.hdata_var1` and `.hdata_var2`')
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
            sc.pp.neighbors(adt, metric=metric, n_neighbors=n_neighbors,
                            use_rep='X')
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
        homonet = pp.normalize_row(self.vv_adj, by='sum')
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
        #            homonet = pp.normalize_row(self.vv_adj, by='sum')
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

        foo = lambda x1, x2: cdist(x1[None, :], x2[None, :], metric=metric)[
            0, 0]
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
        names_freq1 = pp.take_freq1(names1)
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
            self.hdata_var1.obs[key_added] = self.hdata_var.obs[key_added][
                self.var_ids1].values
            self.hdata_var2.obs[key_added] = self.hdata_var.obs[key_added][
                self.var_ids2].values

        params = dict(
            resolution=resolution,
            method=method,
            key_added=key_added,
        )
        if len(kwds) >= 1:
            params.update(kwds)
        self.params['cluster_vars'] = params

        self.extra['var_cluster_counts'] = pp.group_value_counts(
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
            _annos[key_added] = _annos[keys].apply(lambda x: '_'.join(x),
                                                   axis=1)

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

        avgs1 = pp.group_mean(mat1, obs_labels1, features=self.var_names1, )
        avgs2 = pp.group_mean(mat2, obs_labels2, features=self.var_names2, )

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

    def get_module_connection(
            self, ):  # TODO: subset graph --> to_pandas_edgelis
        import networkx as nx
        #        from networkx import to_pandas_edgelist
        g = self.abstracted_graph
        tag1, tag2 = self.params['tags_var']
        edge_df = nx.to_pandas_edgelist(g)
        kept = edge_df[['source', 'target']].apply(
            lambda x: tag1 in x[0] and tag2 in x[1], axis=1)
        return edge_df.loc[kept, :]

    def write_graph(self, fname=None, suffix='.gpickle', resdir=None,
                    write_func=None):
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
            title = 'predicted classes of {} \n({} as the reference)'.format(
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
        ax = uplt.alluvial_plot(
            mat, title=title, cmap_name=cmap, labels=labels,
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
            # fname = 'gene_embeddings_with_private_hlighted.png'
            fp = self.figdir / fname
        else:
            fp = None

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

    def export_abstracted_edges(self, fname=None, resdir=None,
                                tag=None):  # TODO: change
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
