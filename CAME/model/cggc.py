# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 18:59:29 2021

@author: Xingyan Liu
"""

from typing import Union, Sequence, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .base_layers import GeneralRGCLayer  # , HeteroLayerNorm
from .base_layers import BaseMixConvLayer

from .loss import LabelSmoothingCrossEntropy
from .hidden import HiddenRRGCN, HiddenRGCN


# In[]
def detach2numpy(x):
    if isinstance(x, th.Tensor):
        x = x.cpu().clone().detach().numpy()
    return x


# In[]
class CGGCNet(nn.Module):
    '''
    cell-gene-gene-cell network 
    =======================
    Graph Convolutional Network for cell-gene Heterogenous graph, 
    with edges named as:
        ('cell', 'express', 'gene'):        cg_net, 
        ('gene', 'expressed_by', 'cell'):   cg_net.T,
        ('gene', 'homolog_with', 'gene'):   gg_net + sparse.eye(n_gnodes),
        ('cell', 'self_loop_cell', 'cell'):      sparse.eye(n_cells),
        
    * gene embeddings are computed from cells;
    * weight sharing across hidden layers is allowed by setting 
        `share_hidden_weights=True`
    * attention can be applied on the last layer (`self.cell_classifier`);
    * the graph for the embedding layer and the hidden layers can be different;
    * allow expression values as static edge weights. (but seems not work...)

    '''

    def __init__(self,
                 g,
                 in_dim_dict={},
                 h_dim=32,
                 h_dim_add=None,  # None --> rgcn2
                 out_dim=32,  # number of classes
                 num_bases=-1,
                 num_hidden_layers=1,
                 norm='right',
                 use_weight=True,
                 dropout_feat=0.,
                 dropout=0,
                 use_self_loop=False,
                 negative_slope=0.2,
                 batchnorm_ntypes=None,
                 layernorm_ntypes=None,
                 out_bias=False,
                 rel_names_out=None,
                 share_hidden_weights=False,
                 attn_out=True,
                 kwdict_outgat={},
                 share_layernorm=True,
                 residual=False,
                 **kwds):  # ignored
        super(CGGCNet, self).__init__()
        self.in_dim_dict = in_dim_dict
        if h_dim_add is not None:
            if isinstance(h_dim_add, int):
                self.h_dims = (h_dim, h_dim_add)
            elif isinstance(h_dim_add, Sequence):
                self.h_dims = (h_dim,) + tuple(h_dim_add)
        else:
            self.h_dims = (h_dim, h_dim)
        self.out_dim = out_dim
        self.rel_names_out = rel_names_out if rel_names_out is not None else g.etypes
        self.gcn_norm = norm
        self.batchnorm_ntypes = batchnorm_ntypes
        self.layernorm_ntypes = layernorm_ntypes
        self.out_bias = out_bias
        self.attn_out = attn_out

        self.build_embed_layer(activation=nn.LeakyReLU(negative_slope),
                               dropout_feat=dropout_feat,
                               dropout=dropout)

        hidden_model = HiddenRRGCN if share_hidden_weights else HiddenRGCN
        self.rgcn = hidden_model(g.canonical_etypes,
                                 h_dim=h_dim,
                                 out_dim=h_dim_add,  # additional hidden layer if not None
                                 num_hidden_layers=num_hidden_layers,
                                 norm=self.gcn_norm,
                                 use_weight=use_weight,
                                 dropout=dropout,
                                 use_self_loop=use_self_loop,
                                 negative_slope=negative_slope,
                                 batchnorm_ntypes=batchnorm_ntypes,
                                 layernorm_ntypes=layernorm_ntypes,
                                 share_layernorm=share_layernorm,
                                 )

        self.build_cell_classifier(kwdict_gat=kwdict_outgat)
        self.residual = residual

    def forward(self,
                feat_dict, g,
                **kwds):

        if self.residual:
            h_dict0 = self.embed_layer(g, feat_dict, )
            h_dict = self.rgcn.forward(g, h_dict0, norm=True, activate=False, **kwds)
            relu = self.rgcn.leaky_relu
            # residual connection
            h_dict = {'cell': relu(h_dict0['cell'] + h_dict['cell']),
                      'gene': relu(h_dict['gene'])}
        else:
            h_dict = self.embed_layer(g, feat_dict, )
            h_dict = self.rgcn.forward(g, h_dict, **kwds).copy()

        h_dict['cell'] = self.cell_classifier.forward(g, h_dict, **kwds)['cell']

        return h_dict

    def get_hidden_states(self,
                          feat_dict=None, g=None,
                          i_layer=-1,
                          detach2np: bool = True,
                          **kwds):
        '''
        detach2np: whether tensors be detached and transformed into numpy.ndarray
        
        '''
        if (feat_dict is not None) and (g is not None):
            print('Forward passing...')
            # activate the random dropouts, which gives a better integrated embedding
            self.train()
            _ = self.forward(feat_dict, g=g, **kwds)
        else:
            print('No inputs were given for the forward passing, so the '
                  'returned are the current hidden states of the model.')

        h_dict = self.rgcn.hidden_states[i_layer]
        if detach2np:
            h_dict = {
                k: h.cpu().clone().detach().numpy()
                for k, h in h_dict.items()
            }
        return h_dict

    def get_attentions(self, feat_dict, g, fuse='mean'):
        '''
        output:
            cell-by-gene attention matrix
        '''
        #        g = self.g if g is None else g
        # getting subgraph and the hidden states
        g_sub = g.to('cuda')['gene', 'expressed_by', 'cell']
        h_dict = self.get_hidden_states(feat_dict=feat_dict, g=g, detach2np=False)
        # getting heterogenous attention (convolutional) classifier
        HAC = self.cell_classifier.conv.mods['expressed_by']
        feats = (h_dict['gene'], h_dict['cell'])
        HAC.train(False)  # semi-supervised
        _out_dict, attn0 = HAC(g_sub, feats, return_attn=True)

        # constructing attention matrix
        if fuse == 'max':
            attn, _ = th.max(attn0, dim=1)
        else:  # fuse == 'mean':
            attn = th.mean(attn0, dim=1)
        #        else:

        attn = detach2numpy(attn).flatten()
        ig, ic = list(map(detach2numpy, g_sub.edges()))
        n_vnodes, n_obs = g.num_nodes('gene'), g.num_nodes('cell')

        from scipy import sparse
        attn_mat = sparse.coo_matrix(
            (attn, (ig, ic)), shape=(n_vnodes, n_obs)).tocsc().T

        return attn_mat

    def get_classification_loss(
            self, out_cell, labels, weight=None, 
            smooth=True, smooth_eps=0.1, smooth_reduction='mean'):
        # take out representations of nodes that have labels
        # F.cross_entropy() combines `log_softmax` and `nll_loss` in a single function.
        if smooth:
            criterion = LabelSmoothingCrossEntropy(
                eps=smooth_eps, reduction=smooth_reduction)
        else:
            criterion = F.cross_entropy
        class_loss = criterion(out_cell, labels, weight=weight)
        return class_loss

    def build_embed_layer(self, activation=None, **kwds):
        embed_params = dict(
            in_dim_dict=self.in_dim_dict,
            out_dim_dict=self.h_dims[0],
            canonical_etypes=[('cell', 'express', 'gene'),
                              ('cell', 'self_loop_cell', 'cell')],
            norm='right',
            use_weight=True,
            bias=True,  #
            activation=activation,  # None #nn.LeakyReLU(0.2)
            batchnorm_ntypes=self.batchnorm_ntypes,
            layernorm_ntypes=self.layernorm_ntypes,  # None, #
            dropout_feat=0.0,
            dropout=0.2,
            aggregate='sum',
        )
        if len(kwds) > 0:
            embed_params.update(**kwds)
        self.embed_layer = GeneralRGCLayer(**embed_params)

    def build_cell_classifier(self, kwdict_gat={}):
        if self.attn_out:
            self.cell_classifier = self.make_out_gat(kwdict_gat)
        else:
            self.cell_classifier = self.make_out_gcn()

    def make_out_gat(self, kwdict_gat={}):
        mod_params = {
            "gat": dict(
                h_dim=self.out_dim,
                n_heads=8,  # 16, # 16 seems no better
                feat_drop=0.01,
                attn_drop=0.6,
                negative_slope=0.2,
                residual=False,
                activation=None,
                attn_type='add',  # 'mul' or 'add' as the original paper
                heads_fuse='mean',  # 'flat' or 'mean'
            ),
            "gcn": dict(
                norm='right',
                weight=True,
                bias=False,
                activation=None,
            ),
        }
        if len(kwdict_gat) > 0:
            mod_params["gat"].update(kwdict_gat)

        kwdicts = [
            (('gene', 'expressed_by', 'cell'), 'gat', mod_params['gat']),
        ]
        if ('cell', 'self_loop_cell', 'cell') in self.rel_names_out:
            kwdicts.append(
                (('cell', 'self_loop_cell', 'cell'), 'gcn', mod_params['gcn'])
            )

        return BaseMixConvLayer(
            self.h_dims[-1], self.out_dim,
            mod_kwdicts=kwdicts,
            bias=self.out_bias,
            activation=None,
            #                    dropout=self.dropout,
            layernorm_ntypes=['cell'] if self.layernorm_ntypes is not None else None
        )

    def make_out_gcn(self, ):
        return GeneralRGCLayer(
            self.h_dims[0], self.out_dim,
            canonical_etypes=self.rel_names_out,
            norm=self.gcn_norm,
            use_weight=True,
            activation=None,
            bias=self.out_bias,  # if False, No bias in the last layer
            self_loop=False,
            layernorm_ntypes=['cell'])
