# -*- coding: utf-8 -*-
"""
Created on Mon May 17 09:18:41 2021

@author: Xingyan Liu
"""

from pathlib import Path
from typing import Sequence, Union, Mapping, List, Optional
import time
import numpy as np
import torch
from torch import Tensor, LongTensor
import dgl

from ..datapair import AlignedDataPair, DataPair
from .train import BaseTrainer, make_class_weights, seed_everything
from .evaluation import accuracy, get_AMI, get_F1_score
from .plot import plot_records_for_trainer


def prepare4train(
        dpair: Union[DataPair, AlignedDataPair],
        key_class,
        key_clust='clust_lbs',
        scale_within=True,
        unit_var=True,
        clip=False,
        clip_range=(-3, 3.5),
        cluster_labels: Optional[Sequence] = None,
        test_idx: Optional[Sequence] = None,
        test_prop: float = 0.1,
        **kwds  # for code compatibility (not raising error)
) -> dict:
    if isinstance(key_class, str):
        key_class = [key_class] * 2
    if key_clust and cluster_labels is None:
        cluster_labels = dpair.obs_dfs[1][key_clust].values

    feat_dict = dpair.get_feature_dict(scale=scale_within,
                                       unit_var=unit_var,
                                       clip=clip,
                                       clip_range=clip_range)
    labels1, classes1 = dpair.get_obs_labels(
        key_class[0], train_use=0, add_unknown_force=False)
    labels2, classes2 = dpair.get_obs_labels(
        key_class[1], train_use=1, add_unknown_force=False)
    classes1 = classes1[:-1] if 'unknown' in classes1 else classes1
    classes2 = classes2[:-1] if 'unknown' in classes2 else classes2

    n, n1, n2 = dpair.n_obs, dpair.n_obs1, dpair.n_obs2
    if test_idx is None:
        test_idx = (
            np.random.choice(np.arange(n1), int(n1 * test_prop), replace=False),
            np.random.choice(np.arange(n1, n), int(n2 * test_prop), replace=False)
        )
    if isinstance(test_idx[0], int):
        test_idx = (
            [i for i in range(n1) if i in test_idx[0]],
            [i for i in range(n1, n) if i in test_idx[1]]
        )
    train_idx1 = LongTensor([i for i in range(n1) if i not in test_idx[0]])
    train_idx2 = LongTensor([i for i in range(n1, n) if i not in test_idx[1]])
    test_idx1, test_idx2 = tuple(map(LongTensor, test_idx))

    G = dpair.get_whole_net(rebuild=False,)

    ENV_VARs = dict(
        G=G,
        classes=(classes1, classes2),
        feat_dict=feat_dict,
        labels=(labels1, labels2),
        train_idx=(train_idx1, train_idx2),
        test_idx=(test_idx1, test_idx2),
        cluster_labels=cluster_labels,
    )
    return ENV_VARs


class Trainer(BaseTrainer):
    '''
    
    
    '''

    def __init__(self,
                 model,
                 feat_dict: Mapping,
                 g: dgl.DGLGraph,
                 labels: List[Tensor],
                 train_idx: List[Tensor],
                 test_idx: List[Tensor],
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
            l2norm=l2norm,
            use_cuda=use_cuda,
            dir_main=dir_main,
            **kwds
        )
        self.g = g.to(self.device)
        self.set_class_weights()  # class weights
        ### infer n_classes
        self.n_classes1 = len(self.class_weights[0])
        self.n_classes2 = len(self.class_weights[1])

        _record_names = (
            'dur',
            'train_loss1',
            'train_loss',
            'train_acc1',
            'train_acc2',
            'test_acc1',
            'test_acc2',
        )
        self.set_recorder(*_record_names)

    def set_class_weights(
            self, class_weights=None, foo=np.sqrt, n_add=0):
        """
        class_weights: two sequences for corresponding tasks
        """
        if class_weights is None:
            self.class_weights = [
                make_class_weights(
                    self.labels[0][self.train_idx[0]], foo=foo, n_add=n_add),
                make_class_weights(
                    self.labels[1][self.train_idx[1]], foo=foo, n_add=n_add),
            ]
        else:
            self.class_weights = [Tensor(w) for w in class_weights]

        if self.use_cuda:
            self.class_weights = [w.cuda() for w in self.class_weights]

    def plot_class_losses(self, start=0, end=None, fp=None):
        plot_records_for_trainer(
            self,
            record_names=['train_loss1', 'train_loss2'],
            start=start, end=end,
            lbs=['training loss1', 'training loss1'],
            tt='classification losses',
            fp=fp)

    def plot_class_accs(self, start=0, end=None, fp=None):
        plot_records_for_trainer(
            self,
            record_names=['train_acc1', 'test_acc1', 'train_acc2', 'test_acc2'],
            start=start, end=end,
            lbs=['training acc1', 'testing acc1', 'training acc2', 'testing acc2'],
            tt='classification accuracy',
            fp=fp)

    # def plot_cluster_index(self, start=0, end=None, fp=None):
    #     plot_records_for_trainer(
    #         self,
    #         record_names=['test_acc', 'AMI'],
    #         start=start, end=end,
    #         lbs=['test accuracy', 'AMI'],
    #         tt='test accuracy and cluster index',
    #         fp=fp)

    # In[]
    def train(self, n_epochs=350,
              use_class_weights=True,
              params_lossfunc={},
              n_pass=100,
              eps=1e-4,
              cat_class='cell',
              **other_inputs):
        '''
                Main function for model taining 
        ================================================
        
        other_inputs: other inputs for `model.forward()`
        '''
        #        g = self.g
        train_idx1, train_idx2 = self.train_idx
        test_idx1, test_idx2 = self.test_idx
        labels1, labels2 = self.labels

        if use_class_weights:
            class_weights1, class_weights2 = self.class_weights
        else:
            class_weights1, class_weights2 = None, None

        if not hasattr(self, 'acc_max1'):
            self.acc_max1 = 0
        if not hasattr(self, 'acc_max2'):
            self.acc_max2 = 0
        print("start training".center(50, '='))
        self.model.train()

        for epoch in range(n_epochs):
            self._cur_epoch += 1

            self.optimizer.zero_grad()
            t0 = time.time()
            logits = self.model(self.feat_dict,
                                self.g,  # .to(self.device),
                                **other_inputs)
            out_cell = logits[cat_class]
            out_cell1 = out_cell[:, : self.n_classes1]
            out_cell2 = out_cell[:, self.n_classes1:]
            
            loss1 = self.model.get_classification_loss(
                out_cell1[train_idx1, :], labels1[train_idx1],
                weight=class_weights1,
                **params_lossfunc
            )
            loss2 = self.model.get_classification_loss(
                out_cell2[train_idx2, :], labels2[train_idx2],
                weight=class_weights2,
                **params_lossfunc
            )
            loss = loss1 + loss2

            # prediction of ALL
            _, y_pred1 = torch.max(out_cell1, dim=1)
            _, y_pred2 = torch.max(out_cell2, dim=1)

            ### evaluation (Acc.)
            train_acc1 = accuracy(y_pred1[train_idx1], labels1[train_idx1])
            train_acc2 = accuracy(y_pred2[train_idx2], labels2[train_idx2])
            test_acc1 = accuracy(y_pred1[test_idx1], labels1[test_idx1])
            test_acc2 = accuracy(y_pred2[test_idx2], labels2[test_idx2])

            # unsupervised cluster index
            # if self.cluster_labels is not None:
            #     ami = get_AMI(self.cluster_labels, y_pred_test)

            if self._cur_epoch >= n_pass - 1:
                self.acc_max1 = max(self.acc_max1, test_acc1)
                self.acc_max2 = max(self.acc_max2, test_acc2)
                if test_acc1 >= self.acc_max1 - eps or test_acc2 >= self.acc_max2 - eps :
                    self._cur_epoch_best = self._cur_epoch
                    self.save_model_weights()
                    print('model weights backup')

            loss.backward()
            self.optimizer.step()
            t1 = time.time()

            ##########[ recording ]###########
            self._record(dur=t1 - t0,
                         train_loss1=loss1.item(),
                         train_loss2=loss2.item(),
                         train_acc1=train_acc1,
                         train_acc2=train_acc2,
                         test_acc1=test_acc1,
                         test_acc2=test_acc2,
                         )

            dur_avg = np.average(self.dur)
            test_acc_max1 = max(self.test_acc1)
            test_acc_max2 = max(self.test_acc2)
            logfmt = "Epoch {:04d} | Train Acc: ({:.3f}, {:.3f}) | Test Acc: ({:.3f}, {:.3f}) (max=({:.3f}, {:.3f})) | Time: {:.4f}"
            self._cur_log = logfmt.format(
                self._cur_epoch,
                train_acc1, train_acc2,
                test_acc1, test_acc2,
                test_acc_max1, test_acc_max2,
                dur_avg)

            print(self._cur_log)
        self._cur_epoch_adopted = self._cur_epoch

    def eval_current(self,
                     feat_dict=None,
                     g=None,
                     **other_inputs):
        ''' get the current states of the model output
        '''
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


