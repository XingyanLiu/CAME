# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:15:45 2021

@author: Xingyan Liu
"""
from pathlib import Path
import os
from typing import Sequence, Union, Mapping, Optional, List

import time
import random
import numpy as np
from pandas import DataFrame, value_counts
import torch
from torch import Tensor, LongTensor
import dgl
from ..datapair.aligned import AlignedDataPair
from ..datapair.unaligned import DataPair

from ..model import to_device, onehot_encode, multilabel_binary_cross_entropy
from .base import check_dirs, save_json_dict
from .evaluation import accuracy, get_AMI, get_F1_score
from .plot import plot_records_for_trainer


SUBDIR_MODEL = '_models'


def seed_everything(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_class_weights(labels, astensor=True, foo=np.sqrt, n_add=0):
    if isinstance(labels, Tensor):
        labels = labels.cpu().clone().detach().numpy()

    counts = value_counts(labels).sort_index()  # sort for alignment
    n_cls = len(counts) + n_add
    w = counts.apply(lambda x: 1 / foo(x + 1) if x > 0 else 0)  # .tolist() #+ [1 / n_cls]
    #    w  = np.array(w)
    w = (w / w.sum() * (1 - n_add / n_cls)).values
    w = np.array(list(w) + [1 / n_cls] * int(n_add))

    if astensor:
        return Tensor(w)
    return w


def prepare4train(
        dpair: Union[DataPair, AlignedDataPair],
        key_class,
        key_clust='clust_lbs',
        scale_within=True,
        unit_var=True,
        clip=False,
        clip_range=(-3, 3.5),
        categories: Optional[Sequence] = None,
        cluster_labels: Optional[Sequence] = None,
        test_idx: Optional[Sequence] = None,
        ground_truth: bool = True,
        **kwds  # for code compatibility (not raising error)
) -> dict:
    """
    dpair: DataPair
    test_idx: if provided, should be an index-sequence of the same length as 
        `cluster_labels`
    """
    if key_clust and cluster_labels is None:
        cluster_labels = dpair.obs_dfs[1][key_clust].values

    feat_dict = dpair.get_feature_dict(scale=scale_within,
                                       unit_var=unit_var,
                                       clip=clip,
                                       clip_range=clip_range)
    labels, classes = dpair.get_obs_labels(
        key_class, categories=categories, add_unknown_force=False)
    if 'unknown' in classes:
        classes.remove('unknown')

    if test_idx is None:
        train_idx = dpair.get_obs_ids(0, astensor=True)
        test_idx = dpair.get_obs_ids(1, astensor=True)
    else:
        train_idx = LongTensor([i for i in range(dpair.n_obs) if i not in test_idx])
        test_idx = LongTensor(test_idx)

    G = dpair.get_whole_net(rebuild=False, )

    ENV_VARs = dict(
        classes=classes,
        G=G,
        feat_dict=feat_dict,
        train_labels=labels[train_idx],
        test_labels=labels[test_idx] if ground_truth else None,
        train_idx=train_idx,
        test_idx=test_idx,
        cluster_labels=cluster_labels,
    )
    return ENV_VARs


def get_checkpoint_list(dirname):
    all_ckpts = [
        int(_fn.strip('weights_epoch.pt'))
        for _fn in os.listdir(dirname) if _fn.endswith('.pt')
    ]
    return all_ckpts


class BaseTrainer(object):
    """
    DO NOT use it directly!
    
    """

    def __init__(self,
                 model,
                 feat_dict: Mapping,
                 train_labels: Union[Tensor, List[Tensor]],
                 test_labels: Union[Tensor, List[Tensor]],
                 train_idx: Union[Tensor, List[Tensor]],
                 test_idx: Union[Tensor, List[Tensor]],
                 cluster_labels: Optional[Sequence] = None,
                 lr: float = 1e-3,
                 l2norm: float = 1e-2,  # 1e-2 is tested for all datasets
                 dir_main: Union[str, Path] = Path('.'),
                 **kwds  # for code compatibility (not raising error)
                 ):

        # self.use_cuda = use_cuda and torch.cuda.is_available()
        self.set_inputs(
            model, feat_dict,
            train_labels, test_labels,
            train_idx, test_idx,
            )

        self.cluster_labels = cluster_labels
        self.set_train_params(
            lr=lr,
            l2norm=l2norm,  # 1e-2 is tested for all datasets
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=l2norm)

        self.set_dir(dir_main)
        self._cur_epoch = -1
        self._cur_epoch_best = -1
        self._cur_epoch_adopted = 0

    def set_dir(self, dir_main=Path('.')):
        self.dir_main = Path(dir_main)
        self.dir_model = self.dir_main / SUBDIR_MODEL
        check_dirs(self.dir_model)

        print('main directory:', self.dir_main)
        print('model directory:', self.dir_model)

    def set_inputs(
            self, model,
            feat_dict: Mapping,
            train_labels: Union[Tensor, List[Tensor]],
            test_labels: Union[Tensor, List[Tensor], None],
            train_idx: Union[Tensor, List[Tensor]],
            test_idx: Union[Tensor, List[Tensor]],
            ):
        self.model = model
        self.feat_dict = feat_dict
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_idx = train_idx
        self.test_idx = test_idx
        # inference device
        try:
            self.device = self.train_idx.device
        except AttributeError:
            self.device = self.train_idx[0].device

    def all_to_device(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model.to(device)
        self.feat_dict = to_device(self.feat_dict, device)
        self.train_labels = to_device(self.train_labels, device)
        if self.test_labels is not None:
            self.test_labels = to_device(self.test_labels, device)
        self.train_idx = to_device(self.train_idx, device)
        self.test_idx = to_device(self.test_idx, device)
        return (
            self.model,
            self.feat_dict,
            self.train_labels,
            self.test_labels,
            self.train_idx,
            self.test_idx
        )

    def set_train_params(self,
                         lr=1e-3,
                         l2norm=1e-2,  # 1e-2 is tested for all datasets
                         ):
        """
        setting parameters for:
            lr: learning rate
            l2norm: `weight_decay`
        """
        self.lr = lr
        self.l2norm = l2norm

    def set_recorder(self, *names):
        self._record_names = tuple(names)
        for nm in names:
            setattr(self, nm, [])

    def _record(self, **kwds):
        for nm in self._record_names:
            if nm in kwds:
                getattr(self, nm).append(kwds[nm])

    def make_train_logs(self, ):
        dct = {}
        for nm in self._record_names:
            v = getattr(self, nm)
            if len(v) > 0:
                dct[nm] = v
        self._train_logs = DataFrame(dct)

    def write_train_logs(self, fn='train_logs.csv', fp=None):
        if fp is None:
            fp = self.dir_main / fn
        self.make_train_logs()
        self._train_logs.to_csv(fp, index_label='epoch')

    def save_whole_model(self, fn='model.pt', fp=None, **kwds):
        if fp is None:
            fp = self.dir_model / fn
        torch.save(self.model, fp, **kwds)
        print('model saved into:', fp)

    def save_model_weights(self, fp=None, **kwds):
        """ better NOT set the path `fp` manually~
        """
        n_epoch = self._cur_epoch
        if fp is None:
            fp = self.dir_model / f'weights_epoch{n_epoch}.pt'
        torch.save(self.model.state_dict(), fp, **kwds)

    def load_model_weights(self, n_epoch=None, fp=None, **kwds):
        if fp is None:
            if n_epoch is None:
                if self._cur_epoch_best > 0:
                    n_epoch = self._cur_epoch_best
                else:
                    n_epoch = self._cur_epoch
            fp = self.dir_model / f'weights_epoch{n_epoch}.pt'
        sdct = torch.load(fp, **kwds)
        self.model.load_state_dict(sdct)
        self._cur_epoch_adopted = n_epoch
        print('states loaded from:', fp)

    def save_checkpoint_record(self):
        cur_epoch = self._cur_epoch
        if self._cur_epoch_best > 0:
            cur_epoch_rec = self._cur_epoch_best
        else:
            cur_epoch_rec = self._cur_epoch
        all_ckpts = get_checkpoint_list(self.dir_model)
        all_ckpts = [x for x in all_ckpts if
                     x not in {cur_epoch, cur_epoch_rec}]
        ckpt_dict = {
            'recommended': cur_epoch_rec,
            'last': cur_epoch,
            'others': all_ckpts
        }
        save_json_dict(
            ckpt_dict, self.dir_model / 'chckpoint_dict.json')
        # load_json_dict(self.dir_model / 'chckpoint_dict.json')

    def eval_current(self, **other_inputs):
        """ get the current states of the model output
        """
        raise NotImplementedError

    def train(self, ):
        """ abstract method to be re-defined
        """
        raise NotImplementedError

    def train_minibatch(self, **kwargs):
        """ abstract method to be re-defined
        """
        raise NotImplementedError


# In[]

class Trainer(BaseTrainer):
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
                 cluster_labels: Optional[Sequence] = None,
                 lr=1e-3,
                 l2norm=1e-2,  # 1e-2 is tested for all datasets
                 use_cuda=True,
                 dir_main=Path('.'),
                 **kwds  # for code compatibility (not raising error)
                 ):
        super(Trainer, self).__init__(
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

        self.g = g
        self.set_class_weights()
        # infer n_classes
        self.n_classes = len(self.class_weights)
        # for multi-label loss calculation
        self.train_labels_1hot = onehot_encode(
            self.train_labels, sparse_output=False, astensor=True)

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

    def plot_cluster_index(self, start=0, end=None, fp=None):
        plot_records_for_trainer(
            self,
            record_names=['test_acc', 'AMI'],
            start=start, end=end,
            lbs=['test accuracy', 'AMI'],
            tt='test accuracy and cluster index',
            fp=fp)

    def plot_class_accs(self, start=0, end=None, fp=None):
        plot_records_for_trainer(
            self,
            record_names=['train_acc', 'test_acc'],
            start=start, end=end,
            lbs=['training acc', 'testing acc'],
            tt='classification accuracy',
            fp=fp)

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

    # In[]
    def train(self, n_epochs=350,
              use_class_weights=True,
              print_info=True,  # not used
              params_lossfunc={},
              n_pass=100,
              eps=1e-4,
              cat_class='cell',
              device=None,
              backup_stride=43,
              **other_inputs):
        """
                Main function for model training
        ================================================
        
        other_inputs: other inputs for `model.forward()`
        """
        # setting device to train
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.g = self.g.to(device)
        _, feat_dict, train_labels, test_labels, train_idx, test_idx = \
            self.all_to_device(device)
        train_labels_1hot = self.train_labels_1hot.to(device)

        if use_class_weights:
            class_weights = to_device(self.class_weights, device)
        else:
            class_weights = None
        if not hasattr(self, 'ami_max'):
            self.ami_max = 0.
        if not hasattr(self, 'test_acc_max'):
            self.test_acc_max = 0.

        print(f" start training (device='{device}') ".center(60, '='))
        self.model.train()

        for epoch in range(n_epochs):
            self._cur_epoch += 1

            self.optimizer.zero_grad()
            t0 = time.time()
            logits = self.model(self.feat_dict,
                                self.g,
                                **other_inputs)
            out_cell = logits[cat_class]
            loss = self.model.get_classification_loss(
                out_cell[train_idx],
                train_labels,
                weight=class_weights,
                **params_lossfunc
            )
            # multi-label loss
            loss += multilabel_binary_cross_entropy(
                out_cell[train_idx],
                train_labels_1hot,
                weight=class_weights,
            )

            # prediction 
            _, y_pred = torch.max(out_cell, dim=1)
            y_pred_test = y_pred[test_idx]

            # evaluation (Acc.)
            train_acc = accuracy(y_pred[train_idx], train_labels)
            if test_labels is None:
                # ground-truth unavailable
                test_acc = microF1 = macroF1 = weightedF1 = -1.
            else:
                test_acc = accuracy(y_pred_test, test_labels)
                # F1-scores
                microF1 = get_F1_score(test_labels, y_pred_test,
                                       average='micro')
                macroF1 = get_F1_score(test_labels, y_pred_test,
                                       average='macro')
                weightedF1 = get_F1_score(test_labels, y_pred_test,
                                          average='weighted')

            # unsupervised cluster index
            if self.cluster_labels is not None:
                ami = get_AMI(self.cluster_labels, y_pred_test)
            else:
                ami = -1.

            if self._cur_epoch >= n_pass - 1:
                self.ami_max = max(self.ami_max, ami)
                if ami >= self.ami_max - eps:
                    self._cur_epoch_best = self._cur_epoch
                    self.save_model_weights()
                    print('[current best] model weights backup')
                elif self._cur_epoch % backup_stride == 0:
                    self.save_model_weights()
                    print('model weights backup')

            # backward AFTER model saved
            loss.backward()
            self.optimizer.step()
            t1 = time.time()

            #  recording
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
            self.test_acc_max = max(test_acc, self.test_acc_max)
            logfmt = "Epoch {:04d} | Train Acc: {:.4f} | Test Acc or AMI: {:.4f} (max={:.4f}) | AMI={:.4f} | Time: {:.4f}"
            self._cur_log = logfmt.format(
                self._cur_epoch, train_acc,
                ami if test_labels is None else test_acc,
                self.ami_max if test_labels is None else self.test_acc_max,
                ami, dur_avg)
            print(self._cur_log)
        self._cur_epoch_adopted = self._cur_epoch
        self.save_checkpoint_record()

    def eval_current(self,
                     feat_dict=None,
                     g=None,
                     **other_inputs):
        """ get the current states of the model output
        """
        if feat_dict is None:
            feat_dict = self.feat_dict
        if g is None:
            g = self.g
        feat_dict = to_device(feat_dict, self.device)
        g = g.to(self.device)
        with torch.no_grad():
            self.model.train()  # semi-supervised learning
            # self.model.eval()
            output = self.model.forward(feat_dict, g, **other_inputs)
            # output = self.model.get_out_logits(feat_dict, g, **other_inputs)
        return output
