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
from evaluation import detach2numpy
from .train import BaseTrainer, make_class_weights, create_blocks, create_batch, sub_graph,  prepare4train, seed_everything
from .evaluation import accuracy, get_AMI, get_F1_score
from .plot import plot_records_for_trainer

B = shuffle = False


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

    def train_minibatch(self, n_epochs=350,
              use_class_weights=True,
              params_lossfunc={},
              n_pass=100,
              eps=1e-4,
              cat_class='cell',
              batchsize = 128,
              **other_inputs):
        '''
        Funtcion for minibatch trainging
        '''
        train_idx, test_idx, labels = self.train_idx, self.test_idx, self.labels
        #_train_labels, _test_labels = labels[train_idx], labels[test_idx]
        self.g.nodes['cell'].data['feat'] = torch.arange(self.g.num_nodes('cell')).to('cuda')#track the random shuffle
        self.g.nodes['gene'].data['feat'] = torch.arange(self.g.num_nodes('gene')).to('cuda')

        if use_class_weights:
            class_weights = self.class_weights

        if not hasattr(self, 'ami_max'): self.ami_max = 0

        print("start training".center(50, '='))
        self.model.train()
        train_labels, test_labels, batch_list, shuffled_idx = create_batch(train_idx=train_idx,
                                                             test_idx=test_idx,
                                                             batchsize=1024,
                                                             labels=labels,
                                                             shuffle=True)
        n_epochs = 500
        feat_dict = {}
        for epoch in range(n_epochs):
            self._cur_epoch += 1
            all_train_preds = torch.tensor([]).to(self.device)
            all_test_preds = torch.tensor([]).to(self.device)
            t0 = time.time()
            for output_nodes in batch_list:
                blocks = create_blocks(n_layers=3, g=self.g, output_nodes=output_nodes)
                feat_dict['cell'] = self.feat_dict['cell'][blocks[0].nodes['cell'].data['feat'], :]
                batch_train_idx = output_nodes.clone().detach() < len(train_idx)
                batch_test_idx = output_nodes.clone().detach() >= len(train_idx)
                logits = self.model(feat_dict,
                                    blocks,  # .to(self.device),
                                    batch_train = True,
                                    **other_inputs)

                out_cell = logits[cat_class]  # .cuda()
                output_labels = labels[output_nodes]
                out_train_labels = output_labels[batch_train_idx].clone().detach()
                loss = self.model.get_classification_loss(
                    out_cell[batch_train_idx],
                    out_train_labels,
                    weight=class_weights,
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
            # prediction of ALL

            ### evaluation (Acc.)
            train_acc = accuracy(train_labels, all_train_preds)
            test_acc = accuracy(test_labels, all_test_preds)
            t1 = time.time()
            #test_acc = accuracy(y_pred_test, _test_labels)
            #print('epoch', epoch,'loss',loss.item(), 'train_acc', train_acc, 'test_acc', test_acc, 'time', t1-t0)

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
            out_cell = logits[cat_class] # .cuda()
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
