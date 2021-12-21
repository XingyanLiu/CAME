# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:15:45 2021

@author: Xingyan Liu
"""
import logging
from pathlib import Path
import os
from typing import Sequence, Union, Mapping, Optional, List

import time
import random
import numpy as np
from pandas import value_counts
import torch
from torch import Tensor, LongTensor
import dgl
import tqdm
from ..datapair.aligned import AlignedDataPair
from ..datapair.unaligned import DataPair

from ..model import (
    to_device, onehot_encode, idx_hetero, infer_classes,
    multilabel_binary_cross_entropy,
    cross_entropy_loss,
    ce_loss_with_rdrop,
    classification_loss
)
# from ..model._minibatch import create_batch, create_blocks
from .evaluation import accuracy, get_AMI, get_F1_score, detach2numpy
from .plot import plot_records_for_trainer
from ._base_trainer import BaseTrainer, SUBDIR_MODEL


def seed_everything(seed=123):
    """ not works well """
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
    w = counts.apply(lambda x: 1 / foo(x + 1) if x > 0 else 0)  
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
        batch_keys=None,
        unit_var=True,
        clip=False,
        clip_range=(-3, 3.5),
        categories: Optional[Sequence] = None,
        cluster_labels: Optional[Sequence] = None,
        test_idx: Optional[Sequence] = None,
        ground_truth: bool = True,
        node_cls_type: str = 'cell',
        key_label: str = 'label',
        **kwds  # for code compatibility (not raising error)
) -> dict:
    """
    dpair: DataPair
    batch_keys:
        a list of two strings (or None), specifying the batch-keys for
        data1 and data2, respectively.
        If given, features (of cell nodes) will be scaled within each batch
         e.g., ['batch', 'sample']
    test_idx:
        By default, the testing indices will be decided automatically.
        if provided, should be an index-sequence of the same length as
        `cluster_labels`
    """
    if key_clust and cluster_labels is None:
        try:
            cluster_labels = dpair.obs_dfs[1][key_clust].values
        except KeyError:
            logging.warning(
                f"`cluster_labels` is None and `key_clust={key_clust}` is NOT"
                f"found in `dpair.obs_dfs[1].columns`, so not cluster labels"
                f"will be adopted!")

    feat_dict = dpair.get_feature_dict(scale=scale_within,
                                       unit_var=unit_var,
                                       batch_keys=batch_keys,
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

    g = dpair.get_whole_net(rebuild=False, )
    g.nodes[node_cls_type].data[key_label] = labels   # date: 211113

    ENV_VARs = dict(
        classes=classes,
        g=g,
        feat_dict=feat_dict,
        train_labels=labels[train_idx],
        test_labels=labels[test_idx] if ground_truth else None,
        train_idx=train_idx,
        test_idx=test_idx,
        cluster_labels=cluster_labels,
    )
    return ENV_VARs


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
            # use_cuda=use_cuda,
            dir_main=dir_main,
            **kwds  # for code compatibility (not raising error)
        )

        self.g = g
        self.set_class_weights()
        # infer n_classes
        self.n_classes = len(self.class_weights)
        # for multi-label loss calculation
        self.classes = infer_classes(detach2numpy(self.train_labels))  # if classes is None else classes
        # self.train_labels_1hot = onehot_encode(
        #     self.train_labels, sparse_output=False, astensor=True)

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
              params_lossfunc={},
              n_pass=100,
              eps=5e-3,
              cat_class='cell',
              device=None,
              backup_stride: int = 43,
              **other_inputs):
        """ Main function for model training (whole-graph based)

        Parameters
        ----------
        n_epochs: int
            The total number of epochs to train the model
        use_class_weights: bool
            whether to use the class-weights, useful for unbalanced
            sample numbers for each class
        params_lossfunc: dict
            parameters for loss-functions
        n_pass: int
            The number of epochs to be skipped (not backup model checkpoint)
        eps:
            tolerance for cluster-index
        cat_class: str
            node type for classification
        device:
            one of {'cpu', 'gpu', None}
        backup_stride: int
            saving checkpoint after `backup_stride` epochs
        other_inputs:
            other inputs for `model.forward()`

        Returns
        -------
        None
        """
        # setting device to train
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.g = self.g.to(device)
        _, feat_dict, train_labels, test_labels, train_idx, test_idx = \
            self.all_to_device(device)
        # train_labels_1hot = self.train_labels_1hot.to(device)
        train_labels_1hot = onehot_encode(
            self.train_labels, self.classes,
            sparse_output=False, astensor=True).to(device)

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
            logits = self.model(feat_dict, self.g, **other_inputs)[cat_class]
            # logits2 = self.model(feat_dict, self.g, **other_inputs)[cat_class]
            # loss = ce_loss_with_rdrop(
            #     logits, logits2, labels=train_labels,
            #     labels_1hot=train_labels_1hot,
            #     train_idx=train_idx, weight=class_weights,
            #     loss_fn=classification_loss,
            #     **params_lossfunc
            # )
            loss = classification_loss(
                logits[train_idx], train_labels,
                labels_1hot=train_labels_1hot,
                weight=class_weights,
                **params_lossfunc
            )

            # prediction 
            _, y_pred = torch.max(logits, dim=1)
            y_pred_test = y_pred[test_idx]

            # ========== evaluation (Acc.) ==========
            train_acc = accuracy(y_pred[train_idx], train_labels)
            if test_labels is None:
                # ground-truth unavailable
                test_acc = microF1 = macroF1 = weightedF1 = -1.
            else:
                test_acc = accuracy(y_pred_test, test_labels)
                # F1-scores
                microF1 = get_F1_score(test_labels, y_pred_test, 'micro')
                macroF1 = get_F1_score(test_labels, y_pred_test, 'macro')
                weightedF1 = get_F1_score(test_labels, y_pred_test, 'weighted')

            # unsupervised cluster index
            if self.cluster_labels is not None:
                ami = get_AMI(self.cluster_labels, y_pred_test)
            else:
                ami = -1.

            backup = False
            if self._cur_epoch >= n_pass - 1:
                self.ami_max = max(self.ami_max, ami)
                if ami >= self.ami_max - eps > 0:
                    self._cur_epoch_best = self._cur_epoch
                    self.save_model_weights()
                    backup = True
                    print('[current best] model weights backup')
                elif self._cur_epoch % backup_stride == 0:
                    self.save_model_weights()
                    backup = True
                    print('model weights backup')

            # backward AFTER the model being saved
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

            if self.with_ground_truth:
                logfmt = "Epoch {:04d} | Train Acc: {:.4f} | Test: {:.4f} (max={:.4f}) | AMI={:.4f} | Time: {:.4f}"
                self._cur_log = logfmt.format(
                    self._cur_epoch, train_acc, test_acc, self.test_acc_max,
                    ami, dur_avg)
            else:
                logfmt = "Epoch {:04d} | Train Acc: {:.4f} | AMI: {:.4f} (max={:.4f}) | Time: {:.4f}"
                self._cur_log = logfmt.format(
                    self._cur_epoch, train_acc, ami, self.ami_max,
                    ami, dur_avg)

            if self._cur_epoch % 5 == 0 or backup:
                print(self._cur_log)

        self._cur_epoch_adopted = self._cur_epoch
        self.save_checkpoint_record()

    def train_minibatch(self, n_epochs=100,
                        use_class_weights=True,
                        params_lossfunc={},
                        n_pass=100,
                        eps=1e-4,
                        cat_class='cell',
                        batch_size=128,
                        sampler=None,
                        device=None,
                        backup_stride: int = 43,
                        mod_info: int = 3,
                        **other_inputs):
        """ Main function for model training (based on mini-batches)
        """
        # setting device to train
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_idx, test_idx = self.train_idx, self.test_idx
        # train_labels, test_labels = self.train_labels, self.test_labels
        g = self.g
        model = self.model.to(device)
        # self.g.nodes['cell'].data['ids'] = torch.arange(self.g.num_nodes('cell'))  # track the random shuffle
        # self.g.nodes['cell'].data['label'] = labels  # at `prepare4train()`
        if use_class_weights:
            class_weights = to_device(self.class_weights, device)
        else:
            class_weights = None

        if not hasattr(self, 'ami_max'):
            self.ami_max = 0.
        if not hasattr(self, 'test_acc_max'):
            self.test_acc_max = 0.

        if sampler is None:
            sampler = model.get_sampler(g.canonical_etypes, 50)

        train_dataloader = dgl.dataloading.NodeDataLoader(
            # The following arguments are specific to NodeDataLoader.
            g, {'cell': train_idx},  # The node IDs to iterate over in minibatches
            sampler, device=device,  # Put the sampled MFGs on CPU or GPU
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0
        )
        test_dataloader = dgl.dataloading.NodeDataLoader(
            g, {'cell': test_idx}, sampler, device=device, batch_size=batch_size,
            shuffle=False, drop_last=False, num_workers=0
        )
        print(f" start training (device='{device}') ".center(60, '='))
        model.train()
        for epoch in range(n_epochs):
            self._cur_epoch += 1
            all_train_preds = []
            train_labels = []
            t0 = time.time()
            # batch_rank = 0
            # with tqdm.tqdm(train_dataloader) as tq:
            for input_nodes, output_nodes, mfgs in tqdm.tqdm(train_dataloader):
                self.optimizer.zero_grad()
                _labels = mfgs[-1].dstdata['label'][cat_class]
                _feat_dict = to_device(idx_hetero(self.feat_dict, input_nodes), device)
                mfgs = to_device(mfgs, device)
                # _feat_dict = to_device(_feat_dict, device)
                # mfgs = [blk.to(device) for blk in mfgs]

                logits = model(_feat_dict, mfgs, **other_inputs)[cat_class]
                # logits2 = model(_feat_dict, mfgs, **other_inputs)[cat_class]

                # out_train_labels
                out_train_lbs1hot = onehot_encode(
                    _labels, classes=self.classes, astensor=True).to(device)
                _labels = to_device(_labels, device)

                # loss = ce_loss_with_rdrop(
                #     logits, logits2, labels=_labels,
                #     labels_1hot=out_train_lbs1hot,
                #     train_idx=None, weight=class_weights,
                #     loss_fn=classification_loss,
                #     **params_lossfunc
                # )
                loss = classification_loss(
                    logits, _labels,
                    labels_1hot=out_train_lbs1hot,
                    weight=class_weights,
                    **params_lossfunc
                )
                loss.backward()
                self.optimizer.step()

                # _, y_pred = torch.max(logits, dim=1)
                all_train_preds.append(logits.argmax(1).cpu())  # .numpy())
                train_labels.append(_labels.cpu())  # .numpy())
            all_train_preds = torch.cat(all_train_preds, dim=0)
            train_labels = torch.cat(train_labels, dim=0)

            # ========== prediction (test data) ==========
            all_test_preds = []
            test_labels = []
            with tqdm.tqdm(test_dataloader) as tq, torch.no_grad():
                for input_nodes, output_nodes, mfgs in tq:
                    inputs = to_device(idx_hetero(self.feat_dict, input_nodes), device)
                    mfgs = to_device(mfgs, device)
                    # mfgs = [blk.to(device) for blk in mfgs]
                    test_labels.append(
                        mfgs[-1].dstdata['label'][cat_class].cpu())
                    all_test_preds.append(
                        model(inputs, mfgs, )[cat_class].argmax(1).cpu()
                    )
                all_test_preds = torch.cat(all_test_preds, dim=0)
                test_labels = torch.cat(test_labels, dim=0)

            # ========== evaluation (Acc.) ==========
            with torch.no_grad():
                train_acc = accuracy(train_labels, all_train_preds)
                if self.with_ground_truth:
                    test_acc = accuracy(test_labels, all_test_preds)
                    # F1-scores
                    microF1 = get_F1_score(test_labels, all_test_preds, average='micro')
                    macroF1 = get_F1_score(test_labels, all_test_preds, average='macro')
                    weightedF1 = get_F1_score(test_labels, all_test_preds, average='weighted')
                else:
                    test_acc = microF1 = macroF1 = weightedF1 = -1.
                # unsupervised cluster index
                if self.cluster_labels is not None:
                    ami = get_AMI(self.cluster_labels, all_test_preds)
                else:
                    ami = -1.
                backup = False
                if self._cur_epoch >= n_pass - 1:
                    self.ami_max = max(self.ami_max, ami)
                    if ami > self.ami_max - eps > 0:
                        self._cur_epoch_best = self._cur_epoch
                        self.save_model_weights()
                        backup = True
                        print('[current best] model weights backup')
                    elif self._cur_epoch % backup_stride == 0:
                        self.save_model_weights()
                        backup = True
                        print('model weights backup')

                t1 = time.time()

                # recording
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

                if self.with_ground_truth:
                    logfmt = "Epoch {:04d} | Train Acc: {:.4f} | Test: {:.4f} (max={:.4f}) | AMI={:.4f} | Time: {:.4f}"
                    self._cur_log = logfmt.format(
                        self._cur_epoch, train_acc, test_acc, self.test_acc_max,
                        ami, dur_avg)
                else:
                    logfmt = "Epoch {:04d} | Train Acc: {:.4f} | AMI: {:.4f} (max={:.4f}) | Time: {:.4f}"
                    self._cur_log = logfmt.format(
                        self._cur_epoch, train_acc, ami, self.ami_max,
                        ami, dur_avg)

                if self._cur_epoch % mod_info == 0 or backup:
                    print(self._cur_log)

            self._cur_epoch_adopted = self._cur_epoch

    def get_current_outputs(self,
                            feat_dict=None,
                            g=None,
                            batch_size=None,
                            **other_inputs):
        """ get the current states of the model output
        """
        if feat_dict is None:
            feat_dict = self.feat_dict
        if g is None:
            g = self.g
        from ..model import get_model_outputs
        outputs = get_model_outputs(
            self.model, feat_dict, g,
            batch_size=batch_size,
            device=self.device,
            **other_inputs
        )
        return outputs


