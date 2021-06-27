# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:55:09 2021

@author: Xingyan Liu
"""
from pathlib import Path
from typing import Sequence, Union, Mapping, Optional
import time
import numpy as np
import torch
from torch import Tensor
import dgl
from .train import BaseTrainer, make_class_weights, prepare4train, seed_everything
from .evaluation import accuracy, get_AMI, get_F1_score, detach2numpy
from .plot import plot_records_for_trainer
from ..model._utils import to_device

B = shuffle = False


def sub_graph(cell_ID, gene_ID, g):
    """
    Making sub_graph for g with input cell_ID and gene_ID
    """
    output_nodes_dict = {'cell': cell_ID, 'gene': gene_ID}
    g_subgraph = dgl.node_subgraph(g, output_nodes_dict)
    return g_subgraph


def create_blocks(g, output_nodes):
    cell_ID = output_nodes.clone().detach()
    gene_ID = g.in_edges(cell_ID, etype='expressed_by')[0]  # genes expressed_by cells
    gene_ID = torch.unique(gene_ID)
    block = sub_graph(cell_ID, gene_ID, g)  # graph for GAT
    return block


def create_batch(train_idx, test_idx, batchsize, labels, shuffle=True):
    '''
    This function create batch idx, i.e. the cells IDs in a batch.
    ----------------------------------------------------------------------
    Parameters
    -----------
    train_idx: the index for reference cells
    test_idx: the index for query cells
    batchsize: the number of cells in each batch
    labels: the labels for both Reference cells and Query cells
    ----------------------------------------------------------------------
    Returns:
    -----------
    batchlist: the list sores the batch of cell IDs
    all_idx: the shuffled or non-shuffled index for all cells
    train_labels: the shuffled or non-shuffled labels for all reference cells
    test_labels: the shuffled or non-shuffled labels for all query cells
    ----------------------------------------------------------------------
    '''
    batch_list = []
    batch_labels = []
    sample_size = len(train_idx) + len(test_idx)
    if shuffle:
        all_idx = torch.randperm(sample_size)
        shuffled_labels = labels[all_idx]
        train_labels = shuffled_labels[all_idx < len(train_idx)].clone().detach()
        test_labels = shuffled_labels[all_idx >= len(train_idx)].clone().detach()

        if batchsize >= sample_size:
            batch_list.append(all_idx)

        else:
            batch_num = int(len(all_idx) / batchsize) + 1
            for i in range(batch_num - 1):
                batch_list.append(all_idx[batchsize * i: batchsize * (i + 1)])
            batch_list.append(all_idx[batchsize * (batch_num - 1):])

    else:
        train_labels = labels[train_idx].clone().detach()
        test_labels = labels[test_idx].clone().detach()
        all_idx = torch.cat((train_idx, test_idx), 0)
        if batchsize >= sample_size:
            batch_list.append(all_idx)
        else:
            batch_num = int(len(all_idx) / batchsize) + 1
            for i in range(batch_num - 1):
                batch_list.append(all_idx[batchsize * i: batchsize * (i + 1)])
                batch_labels.append(labels[batchsize * i: batchsize * (i + 1)])
            batch_list.append(all_idx[batchsize * (batch_num - 1):])

    return train_labels, test_labels, batch_list, all_idx


class Batch_Trainer(BaseTrainer):
    """


    """

    def __init__(self,
                 model,
                 feat_dict: Mapping,
                 g: dgl.DGLGraph,
                 train_idx: Tensor,
                 test_idx: Tensor,
                 train_labels: Tensor,
                 test_labels: Optional[Tensor] = None,
                 cluster_labels: Union[None, Sequence, Tensor] = None,
                 lr=1e-3,
                 l2norm=1e-2,  # 1e-2 is tested for all datasets
                 use_cuda=True,
                 dir_main=Path('.'),
                 **kwds  # for code compatibility (not raising error)
                 ):
        super(Batch_Trainer, self).__init__(
            model,
            feat_dict=feat_dict,
            train_labels=train_labels,
            test_labels=test_labels,
            train_idx=train_idx,
            test_idx=test_idx,
            cluster_labels=cluster_labels,
            lr=lr,
            l2norm=l2norm,  # 1e-2 is tested for all datasets
            use_cuda=use_cuda,
            dir_main=dir_main,
            **kwds  # for code compatibility (not raising error)
        )
        self.g = g.to(self.device)
        self.set_class_weights()  # class weights
        ### infer n_classes
        self.n_classes = len(self.class_weights)

        _record_names = (
            'dur',
            'train_loss',
            'test_loss',
            'train_acc',
            'test_acc',
            'AMI',
            'microF1',
            'macroF1',
            'weightedF1',
        )
        self.set_recorder(*_record_names)

    def set_class_weights(self, class_weights=None, foo=np.sqrt, n_add=0):
        if class_weights is None:
            self.class_weights = make_class_weights(
                self.train_labels, foo=foo, n_add=n_add)
        else:
            self.class_weights = Tensor(class_weights)

    def plot_class_losses(self, start=0, end=None, fp=None):
        plot_records_for_trainer(
            self,
            record_names=['train_loss', 'test_loss'],
            start=start, end=end,
            lbs=['training loss', 'testing loss'],
            tt='classification losses',
            fp=fp)

    def plot_class_accs(self, start=0, end=None, fp=None):
        plot_records_for_trainer(
            self,
            record_names=['train_acc', 'test_acc'],
            start=start, end=end,
            lbs=['training acc', 'testing acc'],
            tt='classification accuracy',
            fp=fp)

    def plot_cluster_index(self, start=0, end=None, fp=None):
        plot_records_for_trainer(
            self,
            record_names=['test_acc', 'AMI'],
            start=start, end=end,
            lbs=['test accuracy', 'AMI'],
            tt='test accuracy and cluster index',
            fp=fp)

    def train_minibatch(self, n_epochs=100,
                        use_class_weights=True,
                        params_lossfunc={},
                        n_pass=100,
                        eps=1e-4,
                        cat_class='cell',
                        batchsize=128,
                        **other_inputs):
        """
        Funtcion for minibatch trainging
        """
        train_idx, test_idx, train_labels, test_labels = self.train_idx, self.test_idx, self.train_labels, self.test_labels
        labels = torch.cat((train_labels, test_labels), 0)
        self.g.nodes['cell'].data['feat'] = torch.arange(self.g.num_nodes('cell'))  # track the random shuffle

        if use_class_weights:
            class_weights = self.class_weights

        if not hasattr(self, 'ami_max'): self.ami_max = 0
        print("start training".center(50, '='))
        self.model.train()
        self.model = self.model.to('cuda')
        feat_dict = {}
        train_labels, test_labels, batch_list, shuffled_idx = create_batch(train_idx=train_idx,
                                                                           test_idx=test_idx,
                                                                           batchsize=batchsize,
                                                                           labels=labels,
                                                                           shuffle=True)
        for epoch in range(n_epochs):
            self._cur_epoch += 1
            all_train_preds = to_device(torch.tensor([]))
            all_test_preds = to_device(torch.tensor([]))
            t0 = time.time()
            for output_nodes in batch_list:
                block = create_blocks(g=self.g, output_nodes=output_nodes)
                feat_dict['cell'] = self.feat_dict['cell'][block.nodes['cell'].data['feat'], :]
                batch_train_idx = output_nodes.clone().detach() < len(train_idx)
                batch_test_idx = output_nodes.clone().detach() >= len(train_idx)
                logits = self.model(to_device(feat_dict),
                                    to_device(block),  # .to(self.device),
                                    **other_inputs)
                out_cell = logits[cat_class]  # .cuda()
                output_labels = labels[output_nodes]
                out_train_labels = output_labels[batch_train_idx].clone().detach()
                loss = self.model.get_classification_loss(
                    out_cell[batch_train_idx],
                    to_device(out_train_labels),
                    weight=to_device(class_weights),
                    **params_lossfunc
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, y_pred = torch.max(out_cell, dim=1)
                y_pred_train = y_pred[batch_train_idx]
                y_pred_test = y_pred[batch_test_idx]
                all_train_preds = torch.cat((all_train_preds, y_pred_train), 0)
                all_test_preds = torch.cat((all_test_preds, y_pred_test), 0)

            ### evaluation (Acc.)
            with torch.no_grad():
                train_acc = accuracy(train_labels.to('cuda'), all_train_preds)
                test_acc = accuracy(test_labels.to('cuda'), all_test_preds)
                t1 = time.time()
                ### F1-scores
                microF1 = get_F1_score(test_labels, all_test_preds, average='micro')
                macroF1 = get_F1_score(test_labels, all_test_preds, average='macro')
                weightedF1 = get_F1_score(test_labels, all_test_preds, average='weighted')

                ### unsupervised cluster index
                if self.cluster_labels is not None:
                    shuffled_test_idx = detach2numpy(shuffled_idx[shuffled_idx >= len(train_idx)]) - len(train_idx)
                    ami = get_AMI(self.cluster_labels[shuffled_test_idx], all_test_preds)

                if self._cur_epoch >= n_pass - 1:
                    self.ami_max = max(self.ami_max, ami)
                    if ami > self.ami_max - eps:
                        self._cur_epoch_best = self._cur_epoch
                        self.save_model_weights()
                        print('[current best] model weights backup')
                    elif self._cur_epoch % 43 == 0:
                        self.save_model_weights()
                        print('model weights backup')

                t1 = time.time()

                ##########[ recording ]###########
                self._record(dur=t1 - t0,
                             train_loss=loss.item(),
                             train_acc=train_acc,
                             test_acc=test_acc,
                             AMI=ami,
                             microF1=microF1,
                             macroF1=macroF1,
                             weightedF1=weightedF1,
                             )

                dur_avg = np.average(self.dur)
                test_acc_max = max(self.test_acc)
                logfmt = "Epoch {:05d} | Train Acc: {:.4f} | Test Acc: {:.4f} (max={:.4f}) | AMI={:.4f} | Time: {:.4f}"
                self._cur_log = logfmt.format(
                    self._cur_epoch, train_acc,
                    test_acc, test_acc_max,
                    ami, dur_avg)

                print(self._cur_log)
            self._cur_epoch_adopted = self._cur_epoch
