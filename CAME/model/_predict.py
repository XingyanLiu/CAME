# -*- coding: UTF-8 -*-
"""
@CreateDate: 2021/06/21
@Author: Xingyan Liu
@File: _predict.py
@Project: CAME
"""
import logging
from typing import Union, Sequence, Optional
import numpy as np
# from copy import deepcopy
import pandas as pd
import torch
from scipy.special import softmax, expit
from scipy import stats
from . import detach2numpy, onehot_encode
from ..utils.analyze import load_dpair_and_model, load_json_dict, save_json_dict


def sigmoid(x: np.ndarray):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig


def as_probabilities(
        logits,
        mode: str = 'softmax',
):
    """
    Transform model-output logits into probabilities.

    Parameters
    ----------
    logits: np.ndarrary
    mode: str, should either be 'softmax' or 'sigmoid'.
        If 'sigmoid', make multi-label prediction. i.e., predict each class
        independently and each sample could be assigned by more than one class.
        If 'softmax', assume that the classes are mutually exclusive so that
        each sample will be assigned to only one class, with the maximal
        probability.
    Returns
    -------
    np.ndarrary
    """
    x = detach2numpy(logits)
    if mode.lower() == 'softmax':
        return softmax(x, axis=1)
    elif mode.lower() == 'sigmoid':
        return sigmoid(x,)


def predict_from_logits(logits, classes=None):
    """
    logits: shape=(n_sample, n_classes)
    classes: list-like, unique categories
    """
    logits = detach2numpy(logits)
    preds = np.argmax(logits, axis=1)
    if classes is not None:
        preds = np.take(classes, preds)
    return preds


def predict(model: torch.nn.Module,
            feat_dict: dict,
            g=None,
            classes: Sequence = None,
            key: str = 'cell',
            **other_inputs):
    logits = model.forward(feat_dict, g, **other_inputs)[key]
    return predict_from_logits(logits, classes=classes)


def uncertainty_entropy(p, axis=1):
    """ normalized entropy """
    p = np.asarray(p)
    from scipy.stats import entropy
    # the default basis is `e`
    # normalize automatically
    return entropy(p, axis=axis) / np.log(p.shape[axis])


def uncertainty_gini(p, axis=1):
    """ normalized Gini-index """
    p = np.asarray(p)
    sums2 = np.square(p.sum(axis=axis))
    p2sum = np.square(p).sum(axis=axis)
    gini = 1. - p2sum / sums2
    return gini / (1 - 1 / p.shape[axis])


class Predictor(object):
    """
            Unknown class prediction
    =============================================
    Input:
    * output logits of testing samples
    * output logits of training samples

    Output:
    * multi-label probabilities
    * binarized multi-labels
    * uncertainty (p-values)

    Steps:
    0. multi-label probabilities for each sample
    1. fit:
        Compute the background distributions for each class
    2. predict:
        * computing thresholds for each given classes
        * thresholding, reject
        * label the samples with uncertainty
    """

    def __init__(
            self,
            classes: Union[Sequence, int],
            mode: str = 'sigmoid',
            mean: Optional[Sequence] = None,
            std: Optional[Sequence] = None,
    ):
        if isinstance(classes, int):
            self._classes = tuple(range(classes))
        else:
            self._classes = tuple(classes)
        self._mode = mode
        self._background_mean_std = None
        if (mean is not None) and (std is not None):
            self.set_backgrounds(mean, std)

    @property
    def is_fitted(self) -> bool:
        return self._background_mean_std is not None

    @property
    def classes(self) -> tuple:
        return self._classes

    @property
    def n_classes(self) -> int:
        return len(self._classes)

    @classmethod
    def load(cls, json_path, encoding='utf-8'):
        dct = load_json_dict(json_path, encoding=encoding)
        predictor = Predictor(**dct)
        return predictor

    def save(self, json_path, encoding='utf-8'):
        mean, std = list(zip(*self._background_mean_std))
        dct = {
            'classes': list(self.classes),
            'mode': self._mode,
            'mean': list(mean),
            'std': list(std)
        }
        save_json_dict(dct, json_path, encoding=encoding)
        logging.debug(dct)
        logging.info(json_path)

    def set_backgrounds(
            self,
            mean: Sequence,
            std: Sequence
    ):
        # assert len(self.classes) == len(mean)
        self._background_mean_std = tuple(zip(mean, std))

    def fit(self,
            logits: np.ndarray,
            labels: np.ndarray,
            label_encoded: bool = True,
            ):
        """
        Compute the background distributions for each class, i.e., mean and
        standard deviation.

        Parameters
        ----------
        logits
        labels
        label_encoded: bool
            whether the labels are encoded as integers, corresponding to each
            column index

        Returns
        -------
        self, i.e. fitted predictor
        """
        if len(labels.shape) == 1:
            if label_encoded:
                _classes = np.arange(self.n_classes)
            else:
                _classes = self.classes
            labels = onehot_encode(labels, classes=_classes, astensor=False)
        assert logits.shape == labels.shape

        probas = as_probabilities(logits, mode=self._mode)
        # step 1: fit Gaussian distribution of background (negative samples)
        means, stds = [], []
        for i in range(self.n_classes):
            is_this = labels[:, i].toarray().flatten().astype(bool)
            p_bg = probas[~ is_this, i] # TODO: may be empty
            m, std = stats.norm.fit(p_bg)
            means.append(m)
            stds.append(std)
        self.set_backgrounds(means, stds)
        return self

    def decide_thresholds(self, p: float = 0.001, map_class: bool = False):
        if not self.is_fitted:
            raise AttributeError("predictor un-fitted!")
        thresholds = []
        for m, std in self._background_mean_std:
            thresholds.append(stats.norm(m, std).isf(p))
        if map_class:
            thresholds = dict(zip(self.classes, thresholds))
        return thresholds

    def predict(
            self,
            logits,
            p: float = 0.001,
            k: int = 1,
    ) -> np.ndarray:
        probas = as_probabilities(logits, mode=self._mode)
        # decide thresholds and cut-off
        thresholds = self.decide_thresholds(p)
        preds = np.vstack([
            (probas[:, i] > thresholds[i])
            for i in range(self.n_classes)
        ]).astype(int).T

        return preds

    def predict_pvalues(
            self,
            logits,
    ) -> np.ndarray:
        probas = as_probabilities(logits, mode=self._mode)

        pvalues = np.vstack([
            stats.norm(m, std).sf(probas[:, i])
            for i, (m, std) in enumerate(self._background_mean_std)
        ]).T
        return pvalues

    def __str__(self):
        desc = f'''
        Predictor
        - is fitted: {self.is_fitted}
        - classes: {self.classes}
        - backgrounds: {self._background_mean_std}
        '''
        return desc


def __test__():
    test_datadir = "./_temp/('Baron_human', 'Baron_mouse')-(06-20 19.49.07)"
    dpair, model = load_dpair_and_model(test_datadir)
    labels, classes = dpair.get_obs_labels(
        "cell_ontology_class", add_unknown_force=False)
    obs_ids1, obs_ids2 = dpair.obs_ids1, dpair.obs_ids2
    df_logits = pd.read_csv(f'{test_datadir}/df_logits.csv', index_col=0)
    classes = df_logits.columns

    proba = as_probabilities(df_logits.values, mode='sigmoid')

    predictor = Predictor(classes=classes)
    predictor.fit(
        df_logits.values[obs_ids1, :],
        labels[obs_ids1],
    )
    predictor.save(f'{test_datadir}/predictor.json')
    # predictor = Predictor.load(f'{test_datadir}/predictor.json')
    pred_test = predictor.predict(df_logits.values[obs_ids2, :])
    logging.info(f"pred_test {pred_test.shape}:\n{pred_test}")
    pval_test = predictor.predict_pvalues(df_logits.values[obs_ids2, :])
    logging.info(f"pval_test {pval_test.shape}:\n{pval_test}")

    logging.info(predictor)
    return predictor

