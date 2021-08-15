# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:46:19 2021

@author: Xingyan Liu
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        if reduction not in {'sum', 'mean'}:
            raise ValueError('`reduction` should be either "sum" or "mean",'
                             f'got {reduction}')
        self.reduction = reduction

    def forward(self, output, target, weight=None):
        c = output.size()[-1]
        # F.cross_entropy() combines `log_softmax` and `nll_loss` in a single function.
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = - log_preds.sum()
        else:
            loss = - log_preds.sum(dim=-1)
            loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=weight)


def multilabel_binary_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
):
    # probas = F.sigmoid(logits)
    # loss = F.binary_cross_entropy(probas, target, weight=weight)
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    return loss
