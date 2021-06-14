# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:55:09 2021

@author: Xingyan Liu
"""
from pathlib import Path
from typing import Sequence, Union, Mapping
import time
import numpy as np
import torch
from torch import Tensor
import dgl

from .train import BaseTrainer, make_class_weights, create_blocks, \
    create_batch_idx, sub_graph, prepare4train, seed_everything
from .evaluation import accuracy, get_AMI, get_F1_score
from .plot import plot_records_for_trainer


class Trainer(BaseTrainer):
    """
    
    
    """

    def __init__(self,
                 model,
                 feat_dict: Mapping,
                 g: dgl.DGLGraph,
                 labels: Tensor,
                 train_idx: Tensor,
                 test_idx: Tensor,
                 cluster_labels: Union[None, Sequence, Tensor] = None,
                 lr=1e-3,
                 l2norm=1e-2,  # 1e-2 is tested for all datasets
                 use_cuda=True,
                 dir_main=Path('.'),
                 **kwds  # for code compatibility (not raising error)
                 ):
        super(Trainer, self).__init__(
            model,
            feat_dict=feat_dict,
            labels=labels,
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
                self.labels[self.train_idx], foo=foo, n_add=n_add)
        else:
            self.class_weights = Tensor(class_weights)

        if self.use_cuda:
            self.class_weights = self.class_weights.cuda()

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

    def train_minibatch(
            self, n_epochs=350,
            use_class_weights=True,
            params_lossfunc={},
            n_pass=100,
            eps=1e-4,
            cat_class='cell',
            batchsize=128,
            **other_inputs):

        """
        Function for mini-batch training
        """

        train_idx, test_idx, labels = self.train_idx, self.test_idx, self.labels
        _train_labels, _test_labels = labels[train_idx], labels[test_idx]
        # self.g.nodes['cell'].data['feat'] = self.feat_dict['cell'] #{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

        if use_class_weights:
            class_weights = self.class_weights

        if not hasattr(self, 'ami_max'): self.ami_max = 0

        print("start training".center(50, '='))
        self.model.train()
        batch_list, batch_labels = create_batch_idx(train_idx=train_idx,
                                                    test_idx=test_idx,
                                                    batchsize=10455,
                                                    labels=labels, shuffle=True)
        n_epochs = 500
        for epoch in range(n_epochs):
            self._cur_epoch += 1
            all_train_labels = []
            all_test_labels = []
            for output_nodes, output_labels in zip(batch_list, batch_labels):
                blocks = create_blocks(n_layers=3, g=self.g.cpu(),
                                       output_nodes=output_nodes)
                blocks.append(sub_graph(blocks[-1].dstnodes('cell').to('cuda'),
                                        blocks[-1].dstnodes('gene').to('cuda'),
                                        self.g))  # add last block for GAT
                block_batch_train_idx = output_nodes.clone().detach() <= torch.max(
                    train_idx)
                block_batch_test_idx = output_nodes.clone().detach() > torch.max(
                    train_idx)
                self.optimizer.zero_grad()
                t0 = time.time()
                logits = self.model(self.feat_dict,
                                    blocks,  # .to(self.device),
                                    batch_train=True,
                                    **other_inputs)
                out_cell = logits[cat_class]  # .cuda()
                loss = self.model.get_classification_loss(
                    out_cell[block_batch_train_idx],
                    output_labels[block_batch_train_idx],  # labels[train_idx],
                    weight=class_weights,
                    **params_lossfunc
                )
                loss.backward()
                self.optimizer.step()
                _, y_pred = torch.max(out_cell, dim=1)
                y_pred_train = y_pred[block_batch_train_idx]
                y_pred_test = y_pred[block_batch_test_idx]
                # prediction of ALL

                ### evaluation (Acc.)
                if len(output_labels[block_batch_test_idx]) > 0 and len(
                        output_labels[block_batch_train_idx]) > 0:
                    train_acc = accuracy(y_pred[block_batch_train_idx],
                                         output_labels[block_batch_train_idx])
                    test_acc = accuracy(y_pred[block_batch_test_idx],
                                        output_labels[block_batch_test_idx])
                    # test_acc = accuracy(y_pred_test, _test_labels)
                    print('epoch', epoch, 'loss', loss, 'train_acc', train_acc,
                          'test_acc', test_acc)
            '''
            ### F1-scores
            microF1 = get_F1_score(_test_labels, y_pred_test, average='micro')
            macroF1 = get_F1_score(_test_labels, y_pred_test, average='macro')
            weightedF1 = get_F1_score(_test_labels, y_pred_test, average='weighted')

            ### unsupervised cluster index
            if self.cluster_labels is not None:
                ami = get_AMI(self.cluster_labels, y_pred_test)

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
        raise NotImplementedError
        '''

    # In[]
    def train(self, n_epochs=350,
              use_class_weights=True,
              params_lossfunc={},
              n_pass=100,
              eps=1e-4,
              cat_class='cell',
              **other_inputs):
        """
                Main function for model training
        ================================================

        other_inputs: other inputs for `model.forward()`
        """
        #        g = self.g
        train_idx, test_idx, labels = self.train_idx, self.test_idx, self.labels
        _train_labels, _test_labels = labels[train_idx], labels[test_idx]

        if use_class_weights:
            class_weights = self.class_weights

        if not hasattr(self, 'ami_max'): self.ami_max = 0

        print("start training".center(50, '='))
        self.model.train()

        for epoch in range(n_epochs):
            self._cur_epoch += 1

            self.optimizer.zero_grad()
            t0 = time.time()
            logits = self.model(self.feat_dict,
                                self.g,  # .to(self.device),
                                **other_inputs)
            out_cell = logits[cat_class]  # .cuda()
            loss = self.model.get_classification_loss(
                out_cell[train_idx],
                _train_labels,  # labels[train_idx],
                weight=class_weights,
                **params_lossfunc
            )

            # prediction of ALL
            _, y_pred = torch.max(out_cell, dim=1)
            y_pred_test = y_pred[test_idx]

            ### evaluation (Acc.)
            train_acc = accuracy(y_pred[train_idx], labels[train_idx])
            test_acc = accuracy(y_pred_test, _test_labels)
            ### F1-scores
            microF1 = get_F1_score(_test_labels, y_pred_test, average='micro')
            macroF1 = get_F1_score(_test_labels, y_pred_test, average='macro')
            weightedF1 = get_F1_score(_test_labels, y_pred_test,
                                      average='weighted')

            ### unsupervised cluster index
            if self.cluster_labels is not None:
                ami = get_AMI(self.cluster_labels, y_pred_test)

            if self._cur_epoch >= n_pass - 1:
                self.ami_max = max(self.ami_max, ami)
                if ami > self.ami_max - eps:
                    self._cur_epoch_best = self._cur_epoch
                    self.save_model_weights()
                    print('[current best] model weights backup')
                elif self._cur_epoch % 43 == 0:
                    self.save_model_weights()
                    print('model weights backup')

            loss.backward()
            self.optimizer.step()
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

    def eval_current(self,
                     feat_dict=None,
                     g=None,
                     **other_inputs):
        """get the current states of the model output"""

        if feat_dict is None:
            feat_dict = self.feat_dict
        elif self.use_cuda:
            feat_dict = {k: v.cuda() for k, v in feat_dict.items()}
        if g is None:
            g = self.g
        else:
            g = g.to(self.device)
        with torch.no_grad():
            self.model.train()  # semi-supervised learning
            # self.model.eval()
            output = self.model.forward(
                feat_dict, g,  # .to(self.device),
                **other_inputs)
        return output
