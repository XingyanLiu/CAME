# -*- coding: UTF-8 -*-
"""
@CreateDate: 2021/06/21
@Author: Xingyan Liu
@File: _predict.py
@Project: CAME
"""

from typing import Union, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax, expit
from . import detach2numpy


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


# In[]
'''     Unknown class prediction
=============================================
Input:
* output logits of testing samples
* output logits of training samples

Output:
* multi-label probabilities
* binarized multi-labels
* uncertainty

Steps:
0. max multi-label probabilities for each test samples
1. computing thresholds for each given classes
    if references are given, ... else, ...
2. compared with the specific threshold corresponding to that predicted class
3. reject
4. return the index of unknown samples
'''


def predict_probas(logits, mode='sigmoid', only_max=True):
    if mode is None:
        _probas_all = detach2numpy(logits)
    elif mode == 'sigmoid':
        _probas_all = as_probabilities(logits)

    if only_max:
        _probas, _preds = torch.max(_probas_all, dim=1)
        return _probas, _preds
    else:
        return _probas_all  # class membership matrix


def compute_reject_thd(probas_train, preds_train,
                       train_labels,
                       n_classes=None,
                       qt=0.005,
                       thd_min=0, thd_max=0.6,
                       ):
    ''' rejection thresholds for each class
    train_labels: integer labels ranging from 0 to (n_classes - 1)

    '''
    probas_train, preds_train, train_labels = tuple(
        map(detach2numpy, [probas_train, preds_train, train_labels]))
    if n_classes is None:
        n_classes = int(np.max(train_labels))
    #    print(n_classes)
    thresholds = np.zeros(shape=(n_classes,))
    for cl in range(n_classes):
        ind_class_train = train_labels == cl
        ind_class_pred = preds_train == cl
        ind_class_right = np.minimum(ind_class_train, ind_class_pred)
        if ind_class_right.sum() == 0:
            print(
                f'no samples in the training data are rightly labeled as {cl}, skipped')
            continue
        ### NOTE:
        ### only the reightly predicted training samples in the class are considered here
        thd = np.quantile(probas_train[ind_class_right], qt)
        thresholds[cl] = thd
    if thd_max is not None:
        thresholds = np.clip(thresholds, thd_min, thd_max)
        print(
            f'thresholds are clipped, in a range of [{thd_min:.4g}, {thd_max:.4g}]')
    return thresholds


def rejected_idx_from_probas(probas_test, preds_test,
                             thresholds,
                             info=True) -> torch.Tensor:
    '''

    return:
        rejected_idx (ranging from 0 to `len(preds_test)`)
    '''
    rejected_idx = torch.LongTensor()
    for cl, thd in enumerate(thresholds):
        #    cl = classes_code[0] # for testing
        ind_class_test = preds_test == cl
        if ind_class_test.sum() <= 0:
            #        print(f'no samples in the test data are predicted as {cl}, skipped')
            continue
        _is_rejected = torch.min(ind_class_test, probas_test < thd)
        _rejected_idx_cl = torch.nonzero(_is_rejected).flatten()
        n_reject_cl = len(_rejected_idx_cl)
        if info and n_reject_cl > 0:
            print(f'rejection: {n_reject_cl} cells are rejeced by '
                  f'class {cl} (thd={thd:.4f})')
        rejected_idx = torch.cat([rejected_idx, _rejected_idx_cl])

    return rejected_idx


def predict_with_unknown(logits: torch.Tensor,
                         train_idx, train_labels, test_idx,
                         qt=0.005, thd_min=0, thd_max=0.6,
                         n_classes=None, classes=None,
                         info=True, ):
    '''
    logits: shape=(n_samples, n_classes), where n_samples = n_samples1 + n_samples2.
    '''
    n_classes = logits.size()[1] if n_classes is None else n_classes
    ### detach tensors
    _train_idx, _test_idx, _train_labels0 = tuple(
        map(lambda x: x.cpu().clone().detach() if isinstance(x,
                                                             torch.Tensor) else x,
            (train_idx, test_idx, train_labels)))
    probas, preds = predict_probas(logits, softmax=True, only_max=True)
    _probas_train, _preds_train = probas[_train_idx], preds[_train_idx]
    _probas_test, _preds_test = probas[_test_idx], preds[_test_idx]
    #    _probas_test, _preds_test = predict_probas(
    #            logits[_test_idx], softmax=True, only_max=True)

    ### step 1: deciding the thresholds for rejection, based on the
    ### predicted results of the training samples
    thresholds = compute_reject_thd(
        _probas_train, _preds_train,  # logits[_train_idx],
        train_labels,
        n_classes=n_classes,
        qt=qt, thd_min=thd_min, thd_max=thd_max)
    ### step 2: get the sample indices to be rejected from the testing data
    rejected_idx_ = rejected_idx_from_probas(_probas_test, _preds_test,
                                             thresholds, info=info)
    rejected_idx = test_idx[rejected_idx_]
    ### step 3: "unknown" assignmnet (the last class)
    preds[rejected_idx] = n_classes - 1

    if classes is not None:
        return np.take(classes, detach2numpy(preds))
    else:
        return preds.to(logits.device)

