# -*- coding: UTF-8 -*-
"""
@CreateDate: 2021/07/15
@Author: Qunlun Shen
@File: _minibatch.py
@Project: CAME
"""
from pathlib import Path
from typing import Sequence, Union, Mapping, Optional
import time
import numpy as np
import torch
from torch import Tensor
import dgl
import tqdm


def make_fanouts(etypes, etypes_each_layers, k_each_etype: Union[int, dict]):
    if isinstance(k_each_etype, int):
        k_each_etype = dict.fromkeys(etypes, k_each_etype)

    fanouts = []
    for _etypes in etypes_each_layers:
        _fanout = dict.fromkeys(etypes, 0)
        _fanout.update({e: k_each_etype[e] for e in _etypes})
        fanouts.append(_fanout)
    return fanouts


# def idx_hetero(feat_dict, id_dict):
#     sub_feat_dict = {}
#     for k, ids in id_dict.items():
#         if k in feat_dict:
#             sub_feat_dict[k] = feat_dict[k][ids]
#         else:
#             # logging.warning(f'key "{k}" does not exist in {feat_dict.keys()}')
#             pass
#     return sub_feat_dict
#
#
def involved_nodes(g,) -> dict:
    """ collect all the involved nodes from the edges on g
    (a heterogeneous graph)

    Examples
    --------

    >>> input_nodes, output_nodes, mfgs = next(iter(train_dataloader))
    >>> g.subgraph(involved_nodes(mfgs[0]))

    """
    from collections import defaultdict
    nodes = defaultdict(set)
    for stype, etype, dtype in g.canonical_etypes:
        src, dst = g.edges(etype=etype)
        nodes[stype].update(src.numpy())
        nodes[dtype].update(dst.numpy())

    nodes = {k: sorted(v) for k, v in nodes.items()}
    return nodes


def sub_graph(cell_ids, gene_ids, g):
    """
    Making sub_graph for g with input cell_ids and gene_ids
    """
    output_nodes_dict = {'cell': cell_ids, 'gene': gene_ids}
    g_subgraph = dgl.node_subgraph(g, output_nodes_dict)
    return g_subgraph


def create_blocks(g, output_nodes, etype='expressed_by'):
    cell_ids = output_nodes.clone().detach()
    gene_ids = g.in_edges(cell_ids, etype=etype)[0]  # genes expressed_by cells
    gene_ids = torch.unique(gene_ids)
    block = sub_graph(cell_ids, gene_ids, g)  # graph for GAT
    return block


def create_batch(
        sample_size=None,
        train_idx=None,
        test_idx=None,
        batch_size=None,
        labels=None,
        shuffle=True,
        label=True
):
    """
    This function create batch idx, i.e. the cells IDs in a batch.

    Parameters
    ----------
    train_idx:
        the index for reference cells
    test_idx:
        the index for query cells
    batch_size:
        the number of cells in each batch
    labels:
        the labels for both Reference cells and Query cells

    Returns
    -------
    train_labels
        the shuffled or non-shuffled labels for all reference cells
    test_labels
        the shuffled or non-shuffled labels for all query cells
    batch_list
        the list sores the batch of cell IDs
    all_idx
        the shuffled or non-shuffled index for all cells
    """
    if label:
        batch_list = []
        batch_labels = []
        sample_size = len(train_idx) + len(test_idx)
        if shuffle:
            all_idx = torch.randperm(sample_size)
            shuffled_labels = labels[all_idx]
            train_labels = shuffled_labels[all_idx < len(train_idx)].clone().detach()
            test_labels = shuffled_labels[all_idx >= len(train_idx)].clone().detach()

            if batch_size >= sample_size:
                batch_list.append(all_idx)

            else:
                batch_num = int(len(all_idx) / batch_size) + 1
                for i in range(batch_num - 1):
                    batch_list.append(all_idx[batch_size * i: batch_size * (i + 1)])
                batch_list.append(all_idx[batch_size * (batch_num - 1):])

        else:
            train_labels = labels[train_idx].clone().detach()
            test_labels = labels[test_idx].clone().detach()
            all_idx = torch.cat((train_idx, test_idx), 0)
            if batch_size >= sample_size:
                batch_list.append(all_idx)
            else:
                batch_num = int(len(all_idx) / batch_size) + 1
                for i in range(batch_num - 1):
                    batch_list.append(all_idx[batch_size * i: batch_size * (i + 1)])
                    batch_labels.append(labels[batch_size * i: batch_size * (i + 1)])
                batch_list.append(all_idx[batch_size * (batch_num - 1):])

        return train_labels, test_labels, batch_list, all_idx

    else:
        batch_list = []
        if shuffle:
            all_idx = torch.randperm(sample_size)

            if batch_size >= sample_size:
                batch_list.append(all_idx)
            else:
                batch_num = int(len(all_idx) / batch_size) + 1
                for i in range(batch_num - 1):
                    batch_list.append(all_idx[batch_size * i: batch_size * (i + 1)])
                batch_list.append(all_idx[batch_size * (batch_num - 1):])

        else:
            all_idx = torch.arange(sample_size)
            if batch_size >= sample_size:
                batch_list.append(all_idx)
            else:
                batch_num = int(len(all_idx) / batch_size) + 1
                for i in range(batch_num - 1):
                    batch_list.append(all_idx[batch_size * i: batch_size * (i + 1)])
                batch_list.append(all_idx[batch_size * (batch_num - 1):])

        return batch_list, all_idx, None, None


def __test__(g, train_nids):
    k_each_etype = 30

    etypes_each_layers = [
        [('cell', 'express', 'gene'), ('cell', 'self_loop_cell', 'cell')],
        g.canonical_etypes,
        g.canonical_etypes,
        [('gene', 'expressed_by', 'cell')],
    ]

    fanouts = make_fanouts(
        g.canonical_etypes, etypes_each_layers, k_each_etype)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)

    device = 'cpu'
    train_dataloader = dgl.dataloading.NodeDataLoader(
        # The following arguments are specific to NodeDataLoader.
        g,  # The graph
        train_nids,  # The node IDs to iterate over in minibatches
        sampler,  # The neighbor sampler
        device=device,  # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1024,  # Batch size
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0  # Number of sampler processes
    )
