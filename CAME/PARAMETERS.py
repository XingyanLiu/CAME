# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 22:13:17 2021

@author: Xingyan Liu

=================================
    Parameter Settings

* Do NOT change this file directly!

>>> params_pre = PARAMETER.preprocess()
>>> params_model = PARAMETER.model()
>>> params_loss = PARAMETER.loss()

"""
import copy

_params_pre = dict(
    remove_rare=False,  # True for benchmarking; False for case study
    min_samples=10,
    ###
    norm__rev=False,  # False by default
    norm__log_only=False,  # False by default
    ###
    scale_within=True,  # True by default
    unit_var=True,  # True by default
    clip=not True, clip_range=(-3, 5),  # False by default
    ###
    use_degs=True,
    only_1v1homo=False,  # False by default
    target_sum='auto',  # auto --> 1e4
    with_single_vnodes=not True,
)

_params_model = dict(
    h_dim=128,
    num_hidden_layers=2,
    norm='right',
    dropout_feat=0.0,  # dropout for cell input features
    dropout=0.2,
    use_self_loop=False,
    negative_slope=0.05,
    layernorm_ntypes=['cell', 'gene'],
    out_bias=True,
    rel_names_out=[('gene', 'expressed_by', 'cell'),
                   ],
    share_hidden_weights=True,
    attn_out=True,
    kwdict_outgat=dict(n_heads=8,  # 16, # 16 seems no better
                       feat_drop=0.01,
                       attn_drop=0.6,
                       negative_slope=0.2,
                       residual=False,
                       attn_type='add',
                       heads_fuse='mean',
                       ),
    share_layernorm=True,  # ignored if no weights are shared
    residual=not True,  # performance un-tested
)

_params_lossfunc = dict(
    smooth=True, smooth_eps=0.1, smooth_reduction='mean',
    #        smooth=True, smooth_eps=1e-4, smooth_reduction='sum',
)


def get_preprocess_params(**kwds) -> dict:
    params = copy.deepcopy(_params_pre)
    if len(kwds) > 0:
        params.update(**kwds)
    return params


def get_model_params(kwdict_outgat={}, **kwds) -> dict:
    params = copy.deepcopy(_params_model)
    if len(kwds) > 0:
        params.update(**kwds)
    if len(kwdict_outgat) > 0:
        params['kwdict_outgat'].update(kwdict_outgat)
    return params


def get_loss_params(**kwds) -> dict:
    params = copy.deepcopy(_params_lossfunc)
    if len(kwds) > 0:
        params.update(**kwds)
    return params
