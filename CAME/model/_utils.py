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
import dgl
import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from .cggc import CGGCNet
from .cgc import CGCNet
from ._minibatch import create_blocks, create_batch


def detach2numpy(x):
    if isinstance(x, th.Tensor):
        x = x.cpu().clone().detach().numpy()
    elif isinstance(x, Mapping):
        x = {k: detach2numpy(v) for k, v in x.items()}
    elif isinstance(x, List):
        x = [detach2numpy(v) for v in x]
    return x


def to_device(x: Union[th.Tensor, List[th.Tensor], Mapping[Any, th.Tensor], dgl.DGLGraph],
              device='cuda'):
    if not th.cuda.is_available():
        if 'cuda' in str(device):
            logging.warning("`to_device(x)`: CUDA is not available")
        device = 'cpu'
    if isinstance(x, th.Tensor):
        return x.to(device)
    elif isinstance(x, List) and isinstance(x[0], th.Tensor):
        return [xx.to(device) for xx in x]
    elif isinstance(x, Mapping):
        return {k: v.to(device) for k, v in x.items()}
    elif isinstance(x, dgl.DGLGraph):
        return [x.to(device)]
    else:
        raise NotImplementedError('Unresolved input type')


def concat_tensor_dicts(dicts: Sequence[Mapping], dim=0):
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    result = {}
    for _key in list(keys):
        result[_key] = th.cat([d[_key] for d in dicts], dim=dim)
    return result


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
    binarizer = MultiLabelBinarizer(
        classes=classes,
        sparse_output=sparse_output and (not astensor),
    )
    x_onehot = binarizer.fit_transform(x)
    logging.info("classes = %s", binarizer.classes)
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

    # getting heterogeneous attention (convolutional) classifier
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


def get_model_outputs(
        model: nn.Module,
        feat_dict: Mapping,
        g: Union[dgl.DGLGraph, List[dgl.DGLGraph]],
        batch_size: Optional[int] = None,
        device=None,
        **other_inputs
):
    """
    Function facilitate to make mini-batch-wise forward pass

    Parameters
    ----------
    model: heterogeneous graph-neural-network model
    feat_dict: dict of feature matrices
    g: graph or a list or graph (blocks)
    batch_size: int or None
    device:
    other_inputs: other inputs for model.forward function

    Returns
    -------
    model outputs (if mode == 'minibatch', will be merged by batch)
    """

    if batch_size is None:
        if device is not None:
            feat_dict = to_device(feat_dict, device)
            g = g.to(device)
        with th.no_grad():
            model.train()  # semi-supervised learning
            outputs = model.forward(feat_dict, g, **other_inputs)
            # outputs = self.model.get_out_logits(feat_dict, g, **other_inputs)
        return outputs
    else:
        batch_list, all_idx, _, _ = create_batch(
            sample_size=feat_dict['cell'].shape[0],
            batch_size=batch_size, shuffle=False, label=False
        )
        batch_output_list = []
        with th.no_grad():
            for output_nodes in tqdm.tqdm(batch_list):
                model.train()  # semi-supervised learning
                block = create_blocks(g=g, output_nodes=output_nodes)
                _feat_dict = {
                    'cell': feat_dict['cell'][block.nodes['cell'].data['ids'], :]
                }
                if device is not None:
                    _feat_dict = to_device(_feat_dict, device),
                    block = to_device(block, device),
                _out = model.forward(_feat_dict, block, **other_inputs)
                batch_output_list.append(_out)
                # cell_tensor_list.append(_out['cell'])
        # outputs = {'cell': torch.cat(batch_output_list, dim=0)}
        outputs = concat_tensor_dicts(batch_output_list)

    return outputs


