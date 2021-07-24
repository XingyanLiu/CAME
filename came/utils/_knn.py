# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:55:20 2020

@author: Xingyan Liu
"""
from typing import Sequence, Union, Mapping, Iterable, Optional
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scanpy import AnnData
from sklearn.preprocessing import normalize
# from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors  # , BallTree, KDTree


# In[]


def dec_timewrapper(tag='function'):
    def my_decorator(foo):
        def wrapper(*args, **kargs):
            t0 = time.time()
            res = foo(*args, **kargs)
            t1 = time.time()
            print(f'[{tag}] Time used: {t1 - t0: .4f} s')
            return res

        return wrapper

    return my_decorator


def apply_rows(arr, func, **kwds):
    '''
    d: 2-D np.array, function `func` will perform on each row
    -----
    This implementation is a little Faster than the built-in function:
        `np.apply_along_axis(_normalize_dists_single, 1, arr)`
    '''
    if len(kwds) >= 1:
        func = lambda x: func(x, **kwds)
    return np.array(list(map(func, arr)))


def _normalize_dists_1d(d):
    '''
    d: 1-D np.array
    '''
    d1 = d[d.nonzero()]
    vmin = d1.min() * 0.99
    #    vmax = d1.max() * 0.99
    med = np.median(d1)
    d2 = (d - vmin) / (med - vmin)
    d2[d2 < 0] = 0
    return d2


# @dec_timewrapper('manual')
def normalize_dists(d, func=_normalize_dists_1d):
    '''
    d: 2-D np.array, normalization will perform on each row
    -----
    This implementation is a little Faster than the built-in function:
        `np.apply_along_axis(_normalize_dists_single, 1, d)`
    '''
    return apply_rows(d, func)


# @dec_timewrapper('built-in')
def normalize_dists_builtin(d):
    ''' d: 2-D np.array, normalization will perform on each row  '''
    return np.apply_along_axis(_normalize_dists_1d, axis=1, arr=d)


def connect_heat_kernel(d, sigma):
    return np.exp(-np.square(d / sigma))


def _compute_connectivities_umap(
        knn_indices, knn_dists,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
):
    """\
    This is from umap.fuzzy_simplicial_set [McInnes18]_.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """
    n_obs = knn_indices.shape[0]
    n_neighbors = knn_dists.shape[1]
    X = sparse.coo_matrix(([], ([], [])), shape=(n_obs, 1))

    from umap.umap_ import fuzzy_simplicial_set
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    return connectivities.tocsr()


# In[]
''' main function for adata
'''


# In[]

# def stitched_knn(
#        X, group_lbs, 
#        group_order=None,
#        pairs=None,
#        metric='cosine',
#        **kwds):
#    '''
#    group_lbs: 
#        list-like, np.array; shape = (n_samples,)
#    '''
#    pass

def find_neighbors(X, n_neighbors=8, metric='cosine',
                   fuzzynorm=True,
                   algorithm=None, **kwds
                   ):
    '''
    X: shape=(n_samples, n_vars)
    algorithm='brute' means exact KNN search by default
    
    '''
    if algorithm is None:
        algorithm = 'brute' if X.shape[0] <= 3000 else 'auto'
    indexer = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm=algorithm, metric=metric,
        **kwds).fit(X)
    dists, inds = indexer.kneighbors(return_distance=True)
    dist_mat = _sparse_mat_from_inds_dists(inds, dists, )

    if fuzzynorm:
        conn = _compute_connectivities_umap(inds, dists)
        return dist_mat, conn
    else:
        return dist_mat


def pair_stitched_knn(
        X1, X2,
        ks=10,
        ks_inner=5,
        metric='cosine',
        func_norm=None,
        algorithm='auto',
        metric_params: Mapping = None,
        **kwds):
    ''' Mutrual KNN seaching for a pair of datasets.
   
    X1, X2: 
        np.array; 
        shape = (n_samples1, n_vars) and (n_samples2, n_vars)
        the data matrix, with each row as a sample.
        
    ks:
        samples in X1 will search ks[0] NNs in X2, samples in X2 will 
        search ks[1] NNs in X1.
        
    returns
    =======
    distances, connect: 
        sparse.csr_matrix
        
    Example:
    >>> distances, connect = pair_stitched_knn(X[ids1, :], X[ids2, :])
        
    '''
    N1 = X1.shape[0]
    N2 = X2.shape[1]

    if not isinstance(ks, Iterable):
        ks = [ks] * 2
    if not isinstance(ks_inner, Iterable):
        ks_inner = [ks_inner] * 2

    if metric == 'cosine':
        # pretended cosine matric by L2-normalization
        X1, X2 = tuple(map(lambda x: normalize(x, axis=1), (X1, X2)))

    indexer1 = NearestNeighbors(
        n_neighbors=ks[0],
        algorithm=algorithm,
        metric=metric, metric_params=metric_params,
        **kwds).fit(X1)
    indexer2 = NearestNeighbors(
        n_neighbors=ks[1],
        algorithm=algorithm,
        metric=metric, metric_params=metric_params,
        **kwds).fit(X2)

    print('1 in 1')
    if ks_inner[0] >= 1:
        dists11, _11 = _indexing_knn_graph(
            indexer1, None, k=ks_inner[0], func_norm=func_norm)
    else:
        dists11 = sparse.csr_matrix((N1, N1))
    print('2 in 2')
    if ks_inner[0] >= 1:
        dists22, _22 = _indexing_knn_graph(
            indexer2, None, k=ks_inner[1], func_norm=func_norm)
    else:
        dists22 = sparse.csr_matrix((N2, N2))
    print('1 in 2')
    dists12, _12 = _indexing_knn_graph(
        indexer2, X1, k=ks_inner[0], func_norm=func_norm)
    print('2 in 1')
    dists21, _21 = _indexing_knn_graph(
        indexer1, X2, k=ks_inner[1], func_norm=func_norm)
    ''' What it does?
    '''
    adj_bin12 = _binarize_mat(dists12)
    adj_bin21 = _binarize_mat(dists21)
    mnn12 = adj_bin12.multiply(adj_bin21.T)
    dists12 = dists12.multiply(mnn12)
    dists21 = dists21.multiply(mnn12.T)

    print('combining distances')
    distances = _concat_sparse_mats([dists11, dists12], [dists21, dists22])
    if func_norm is None:
        connect = distances.copy()
    #        connect[connect > 0] = 1
    else:
        connect = _concat_sparse_mats([_11, _12], [_21, _22])
    #    print(type(distances)) # COO
    print('Done!')
    distances += distances.T  # TODO
    connect += connect.T
    return distances.tocsr(), connect.tocsr()


def _binarize_mat(mat, inplace=False):
    if not inplace:
        mat = mat.copy()
    mat[mat > 0] = 1
    return mat


def _concat_sparse_mats(*mat_lists):
    '''
    The following code are equivalence:
        
    >>> _concat_sparse_mats(*[[dists11, dists12], [dists21, dists22]])
    
    >>> _concat_sparse_mats([dists11, dists12], [dists21, dists22])
    
    >>> dists1_ = sparse.hstack([dists11, dists12])
    ... dists2_ = sparse.hstack([dists21, dists22])
    ... distances = sparse.vstack([dists1_, dists2_])
    '''
    mat_rows = list(map(sparse.hstack, mat_lists))
    cated_mat = sparse.vstack(mat_rows)
    return cated_mat  # COO


def _indexing_knn_graph(indexer,
                        X=None,
                        k=None,
                        func_norm='umap',
                        **kwds):
    '''
    
    returns:
        A sparse graph in CSR format, shape = [n_queries, n_samples_fit]
        
    Example:
    >>> dists = _indexing_knn_graph(indexer, X, k)
    >>> dists, dists_n = _indexing_knn_graph(indexer, X, k, func_norm)
    '''
    if func_norm is None:
        dist_graph = indexer.kneighbors_graph(X, n_neighbors=k, mode='distance')
        return dist_graph, None

    else:
        knn_dists, knn_indices = indexer.kneighbors(
            X, n_neighbors=k, return_distance=True)

        # construction of the sparse distance matrix
        N2 = indexer.n_samples_fit_
        dist_graph = _sparse_mat_from_inds_dists(
            knn_indices, knn_dists, n_cols=N2)

        if func_norm == 'umap':
            connectivities = _compute_connectivities_umap(  # BUG!
                knn_indices, knn_dists, )  # a symetric sparse matrix
            return dist_graph, connectivities

        if callable(func_norm):
            knn_dists_normed = apply_rows(knn_dists, func_norm)
            dist_graph_normed = _sparse_mat_from_inds_dists(
                knn_indices, knn_dists_normed, n_cols=N2)
            return dist_graph, dist_graph_normed

        else:
            raise ValueError('`func_norm` should be one of {None, "umap", callable}, '
                             f'got {func_norm}')


def _sparse_mat_from_inds_dists(knn_indices, knn_dists,
                                n_cols=None,
                                cut_negative=True):
    n_rows = knn_indices.shape[0]
    n_cols = n_rows if n_cols is None else n_cols
    k = knn_dists.shape[1]
    inds1 = np.repeat(np.arange(n_rows), k)
    inds2 = knn_indices.flatten()
    if cut_negative:
        knn_indices[knn_indices < 0] = 0.0

    sparse_mat = sparse.coo_matrix(
        (knn_dists.flatten(), (inds1, inds2)),
        shape=(n_rows, n_cols))
    sparse_mat.eliminate_zeros()

    return sparse_mat.tocsr()


if __name__ == '__main__':
    x = np.arange(2400).reshape(-1, 50)
    x_norm1 = normalize_dists_builtin(x)
    x_norm2 = normalize_dists(x)
    print(x.shape)
    print(x_norm1)
    #    print(x_norm2)

    ''' TEST: _concat_sparse_mats '''

    x = sparse.eye(5)
    res = _concat_sparse_mats([x, x * 2, x * 3], [x * 4, x * 5, x * 6])
    print(res.toarray())
