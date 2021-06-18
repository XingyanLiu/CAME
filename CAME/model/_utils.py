# -*- coding: UTF-8 -*-
"""
@author: Xingyan Liu
@file: utils.py
@time: 2021-06-12
"""

from typing import Union, Sequence, Optional, Mapping, Any, List
import logging
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.preprocessing import MultiLabelBinarizer
from .cggc import CGGCNet
from .cgc import CGCNet


def detach2numpy(x):
    if isinstance(x, th.Tensor):
        x = x.cpu().clone().detach().numpy()
    elif isinstance(x, Mapping):
        x = {k: detach2numpy(v) for k, v in x.items()}
    return x


def to_cuda(x):
    if isinstance(x, th.Tensor):
        return x.cuda()
    elif isinstance(x, List) and isinstance(x[0], th.Tensor):
        return [xx.cuda() for xx in x]
    elif isinstance(x, Mapping):
        return {k: v.cuda() for k, v in x.items()}


def onehot_encode(
        x: Sequence,
        classes=None,
        sparse_output: bool = True,
        astensor: bool = True,
        **kwargs
):
    x = detach2numpy(x)
    if not isinstance(x[0], Sequence):
        x = [[_x] for _x in x]
    binarizer = MultiLabelBinarizer(classes=classes, sparse_output=sparse_output)
    x_onehot = binarizer.fit_transform(x)
    print(binarizer.classes)
    if astensor:
        return th.Tensor(x_onehot)
    else:
        return x_onehot


def get_all_hidden_states(
        model: Union[CGGCNet, CGCNet, nn.Module],
        feat_dict: Mapping[Any, th.Tensor],
        g: dgl.DGLGraph,
        detach2np: bool = True,
):
    # embedding layer
    h_embed = model.embed_layer(g, feat_dict)
    # hidden layers
    _ = model.rgcn(g, h_embed)
    h_list = [h_embed] + model.rgcn.hidden_states
    if detach2np:
        h_list = [detach2numpy(h) for h in h_list]
    return h_list


def get_attentions(
        model: nn.Module,
        feat_dict: Mapping,
        g: dgl.DGLGraph,
        fuse='mean',
        from_scratch: bool = True,
):
    """
    output:
        cell-by-gene attention matrix (sparse)
    """
    if from_scratch:
        h_dict = model.get_hidden_states(
            feat_dict=feat_dict, g=g, detach2np=False)
    else:
        h_dict = feat_dict

    # getting subgraph and the hidden states
    g_sub = g.to('cuda')['gene', 'expressed_by', 'cell']

    # getting heterogenous attention (convolutional) classifier
    HAC = model.cell_classifier.conv.mods['expressed_by']
    feats = (h_dict['gene'], h_dict['cell'])
    HAC.train(False)  # semi-supervised
    _out_dict, attn0 = HAC(g_sub, feats, return_attn=True)

    # constructing attention matrix
    if fuse == 'max':
        attn, _ = th.max(attn0, dim=1)
    elif fuse == 'mean':
        attn = th.mean(attn0, dim=1)
    else:
        raise ValueError("`fuse` should be either 'max' or 'mean'")

    attn = detach2numpy(attn).flatten()
    ig, ic = list(map(detach2numpy, g_sub.edges()))
    n_vnodes, n_obs = g.num_nodes('gene'), g.num_nodes('cell')
    from scipy import sparse
    attn_mat = sparse.coo_matrix(
        (attn, (ig, ic)), shape=(n_vnodes, n_obs)).tocsc().T

    return attn_mat


