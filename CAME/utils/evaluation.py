# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:43:10 2021

@author: Xingyan Liu
"""

import numpy as np
from scipy.special import softmax
from sklearn import metrics
import torch
from typing import Sequence

def detach2numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().clone().detach().numpy()
    return x

def accuracy(logits, labels):
    if len(logits.shape) >= 2:
        _, preds = torch.max(logits, dim=1)
    else:
        preds = logits
    correct = torch.sum(preds == labels)
    return correct.item() * 1.0 / len(labels)


def get_AMI(y_true, y_pred, **kwds):
    y_true, y_pred = list(map(
            detach2numpy, 
            (y_true, y_pred)))
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred, **kwds)
    return ami

def get_F1_score(y_true, y_pred, average='micro', **kwds):
    y_true, y_pred = list(map(detach2numpy, (y_true, y_pred)))
    f1 = metrics.f1_score(y_true, y_pred, average=average, **kwds)
    return f1


def as_probabilities(logits):
    return softmax(detach2numpy(logits), axis=1)

def predict_from_logits(logits, classes=None):
    '''
    logits: shape=(n_sample, n_classes)
    classes: list-like, unique categories
    '''
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
    
    

