# -*- coding: UTF-8 -*-
"""
@Author: Xingyan Liu
@File: _base_trainer.py
@Date: 2021-08-22
@Project: CAME
"""
import os
from pathlib import Path
from typing import Union, Optional, Sequence, Mapping, List
import logging
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from .base import check_dirs, save_json_dict
from ..model import to_device, onehot_encode, multilabel_binary_cross_entropy

SUBDIR_MODEL = '_models'


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
        self.model = None
        self.feat_dict = None
        self.train_labels = None
        self.test_labels = None
        self.train_idx = None
        self.test_idx = None
        self.device = None
        self.dir_main = None
        self.dir_model = None
        self.lr = None
        self.l2norm = None
        self._record_names = None
        self._train_logs = None
        self.with_ground_truth = None

        self.set_dir(dir_main)
        self.set_inputs(
            model, feat_dict,
            train_labels, test_labels,
            train_idx, test_idx,
        )

        self.cluster_labels = cluster_labels
        self._set_train_params(
            lr=lr,
            l2norm=l2norm,
        )

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
            self, model: torch.nn.Module,
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
        self.with_ground_truth = self.test_labels is not None
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

    def _set_train_params(self,
                          lr: float = 1e-3,
                          l2norm: float = 1e-2,
                          ):
        """
        setting parameters for model training

        Parameters
        ----------
        lr: float
            the learning rate (default=1e-3)
        l2norm:
            the ``weight_decay``, 1e-2 is tested for all datasets
        """
        self.lr = lr
        self.l2norm = l2norm
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=l2norm)

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
        self._train_logs = pd.DataFrame(dct)

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
            ckpt_dict, self.dir_model / 'checkpoint_dict.json')
        # load_json_dict(self.dir_model / 'checkpoint_dict.json')

    def get_current_outputs(self, **other_inputs):
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


def __test__():
    pass


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')
    __test__()
