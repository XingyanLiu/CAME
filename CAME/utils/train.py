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
import torch
from ..datapair.aligned import AlignedDataPair
from ..datapair.unaligned import DataPair
from .base import check_dirs
from .evaluation import accuracy, get_AMI
from .plot import plot_records_for_trainer


'''
def seach_connected_cell_gene_ID(cell_ID, gene_ID, g):
    cell_conne_with_gene = torch.unique(g.in_edges(gene_ID, etype='express')[0])
    cell_conne_with_cell = torch.unique(g.in_edges(cell_ID, etype='similar_to')[0])
    all_cell_ID = torch.cat((cell_ID, cell_conne_with_gene, cell_conne_with_cell), axis=0)
    all_cell_ID = torch.unique(all_cell_ID)
    
    gene_connce_with_gene = torch.unique(g.in_edges(gene_ID, etype='homolog_with')[0])
    gene_connce_with_cell = torch.unique(g.in_edges(cell_ID, etype='expressed_by')[0])
    all_gene_ID = torch.cat((gene_connce_with_gene, gene_connce_with_cell), axis=0)
    all_gene_ID = torch.unique(all_gene_ID)
    
    return all_cell_ID, all_gene_ID
'''

def sub_graph(cell_ID, gene_ID, g):
	###sub_graph for g with input cell_ID and gene_ID
	output_nodes_dict = {'cell': cell_ID, 'gene': gene_ID}
	g_subgraph = dgl.node_subgraph(g, output_nodes_dict)

	return g_subgraph


def create_blocks(n_layers, g, output_nodes):
    #blocks = []
    cell_ID = output_nodes.clone().detach()
    gene_ID = g.in_edges(cell_ID, etype='expressed_by')[0]#genes expressed_by cells
    gene_ID = torch.unique(gene_ID)
    block = sub_graph(cell_ID, gene_ID, g) # graph for GAT
    blocks = [block]
    for i in range(n_layers):
        output_nodes_dict = {'cell': cell_ID, 'gene' : gene_ID}
        frontier = dgl.in_subgraph(g, output_nodes_dict)
        #block = dgl.to_block(frontier, output_nodes_dict)
        blocks.append(block)
        cell_ID = block.nodes['cell'].data['feat']
        gene_ID = block.nodes['gene'].data['feat']
        #all_cell_ID, all_gene_ID = seach_connected_cell_gene_ID(cell_ID, gene_ID, g)
    blocks.reverse() 

    return blocks


def create_batch(train_idx, test_idx, batchsize, labels, shuffle=True):
    '''
    This function create batch idx, i.e. the cells IDs in a batch.
    ########################################################################
    all_idx
    type: array
    value: stores all the cells' IDs , i.e. array([0, 1, 2,..., n])
    ########################################################################
    return: batchlist, which is on gpu with a shuffle.
    ########################################################################
    '''
    batch_list = []
    batch_labels = []
    sample_size = len(train_idx) + len(test_idx)
    #all_idx = torch.cat((train_idx, test_idx), 0)
    if shuffle:
        all_idx = torch.randperm(sample_size).to('cuda')
        shuffled_labels = labels[all_idx] ###copy but not replace
        train_labels = shuffled_labels[all_idx < len(train_idx)].clone().detach()
        test_labels = shuffled_labels[all_idx >= len(train_idx)].clone().detach()

        if batchsize >= sample_size:
            batch_list.append(all_idx)

        else:
            batch_num = int(len(all_idx) / batchsize) + 1
            for i in range(batch_num-1):
                batch_list.append(all_idx[batchsize*i: batchsize*(i+1)])
            batch_list.append(all_idx[batchsize*(batch_num-1): ])

    else:
        train_labels = labels[train_idx].clone().detach()
        test_labels = labels[test_idx].clone().detach()
        all_idx = torch.cat((train_idx, test_idx), 0)
        if batchsize >= (sample_size):
            batch_list.append(all_idx)
            #batch_labels.append(labels)
        else:
            batch_num = int(len(all_idx) / batchsize) + 1
            for i in range(batch_num-1):
                batch_list.append(all_idx[batchsize*i: batchsize*(i+1)])
                batch_labels.append(labels[batchsize*i: batchsize*(i+1)])
            batch_list.append(all_idx[batchsize*(batch_num-1): ])

    return train_labels, test_labels, batch_list, all_idx



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
        **kwds  # for code compatibility (not raising error)
) -> dict:
    """
    dpair: DataPair
    test_idx: if provided, should be an index-sequence of the same length as 
        `cluster_labels`
    """
    if key_clust and cluster_labels is None:
        cluster_labels = dpair.obs_dfs[1][key_clust].values
    # else:
    #     cluster_labels = None

    feat_dict = dpair.get_feature_dict(scale=scale_within,
                                       unit_var=unit_var,
                                       clip=clip,
                                       clip_range=clip_range)
    labels0, classes = dpair.get_obs_labels(
        key_class, categories=categories, add_unknown_force=False)
    classes = classes[:-1] if 'unknown' in classes else classes

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
        labels=labels0,
        train_idx=train_idx,
        test_idx=test_idx,
        cluster_labels=cluster_labels,
    )
    return ENV_VARs


# In[]

class BaseTrainer(object):
    """
    DO NOT use it directly!
    
    """

    def __init__(self,
                 model,
                 feat_dict: Mapping,
                 labels: Union[Tensor, List[Tensor]],
                 train_idx: Union[Tensor, List[Tensor]],
                 test_idx: Union[Tensor, List[Tensor]],
                 cluster_labels: Optional[Sequence] = None,
                 lr: float = 1e-3,
                 l2norm: float = 1e-2,  # 1e-2 is tested for all datasets
                 use_cuda: bool = True,
                 dir_main: Union[str, Path] = Path('.'),
                 **kwds  # for code compatibility (not raising error)
                 ):

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.set_inputs(model, feat_dict,
                        labels, train_idx, test_idx,
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
        self.dir_model = self.dir_main / '_models'
        check_dirs(self.dir_model)

        print('main directory:', self.dir_main)
        print('model directory:', self.dir_model)

    def set_inputs(self, model,
                   feat_dict: Mapping,
                   labels: Union[Tensor, List[Tensor]],
                   train_idx: Union[Tensor, List[Tensor]],
                   test_idx: Union[Tensor, List[Tensor]],
                   ):

        if self.use_cuda:
            def _to_cuda(x):
                if isinstance(x, Tensor):
                    return x.cuda()
                elif isinstance(x[0], Tensor):
                    return [xx.cuda() for xx in x]
            print('Using CUDA...')
            model.cuda()
            feat_dict = {k: feat_dict[k].cuda() for k in feat_dict.keys()}
            labels = _to_cuda(labels)
            train_idx = _to_cuda(train_idx)
            test_idx = _to_cuda(test_idx)

        self.model = model
        self.feat_dict = feat_dict
        self.labels = labels
        self.train_idx = train_idx
        self.test_idx = test_idx
        # inference device
        try:
            self.device = self.train_idx.device
        except AttributeError:
            self.device = self.train_idx[0].device

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
        #        dct = {nm: getattr(self, nm) for nm in self._record_names}
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
                 labels: Tensor,
                 train_idx: Tensor,
                 test_idx: Tensor,
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
            labels=labels,
            train_idx=train_idx,
            test_idx=test_idx,
            cluster_labels=cluster_labels,
            lr=lr,
            l2norm=l2norm,  # 1e-2 is tested for all datasets
            use_cuda=use_cuda,
            dir_main=dir_main,
            **kwds  # for code compatibility (not raising error)
        )

        self.g = g.to(self.device)
        self.set_class_weights()
        ### infer n_classes
        self.n_classes = len(self.class_weights)

        _record_names = (
            'dur',
            'train_loss',
            'train_acc',
            'AMI',
        )
        self.set_recorder(*_record_names)

    def set_class_weights(self, class_weights=None, foo=np.sqrt, n_add=0):
        if class_weights is None:
            self.class_weights = make_class_weights(
                self.labels[self.train_idx], foo=foo, n_add=n_add)
        else:
            self.class_weights = Tensor(class_weights)
        if self.use_cuda:
            self.class_weights = self.class_weights.cuda()

    def plot_cluster_index(self, start=0, end=None, fp=None):
        plot_records_for_trainer(
            self,
            record_names=['train_acc', 'AMI'],
            start=start, end=end,
            lbs=['train accuracy', 'AMI'],
            tt='train accuracy and cluster index',
            fp=fp)

    def train_minibatch(self, **kwargs):
        # TODO: @qunlun
        raise NotImplementedError

    # In[]
    def train(self, n_epochs=350,
              use_class_weights=True,
              print_info=True,  # not used
              params_lossfunc={},
              n_pass=100,
              eps=1e-4,
              cat_class='cell',
              **other_inputs):
        """
                Main function for model training
        ================================================
        
        other_inputs: other inputs for `model.forward()`
        """
        train_idx, test_idx, labels = self.train_idx, self.test_idx, self.labels
        _train_labels, _test_labels = labels[train_idx], labels[test_idx]

        if use_class_weights:
            class_weights = self.class_weights

        print("start training".center(50, '='))
        self.model.train()
        _train_labels = self.labels[train_idx]  # might be changed during AL

        for epoch in range(n_epochs):
            self._cur_epoch += 1

            self.optimizer.zero_grad()
            t0 = time.time()
            logits = self.model(self.feat_dict,
                                self.g,  # .to(self.device),
                                **other_inputs)
            out_cell = logits[cat_class]
            loss = self.model.get_classification_loss(
                out_cell[train_idx],
                _train_labels,  # labels[train_idx],
                weight=class_weights,
                **params_lossfunc
            )

            # prediction 
            _, y_pred = torch.max(out_cell, dim=1)
            y_pred_test = y_pred[test_idx]

            ### evaluation (Acc.)
            train_acc = accuracy(y_pred[train_idx], labels[train_idx])

            ### unsupervised cluster index
            if self.cluster_labels is not None:
                ami = get_AMI(self.cluster_labels, y_pred_test)

            if self._cur_epoch >= n_pass - 1:
                ami_max = max(self.AMI[n_pass - 1:])
                if ami >= ami_max - eps:
                    self._cur_epoch_best = self._cur_epoch
                    self.save_model_weights()
                    print('[current best] model weights backup')
                elif self._cur_epoch % 43 == 0:
                    self.save_model_weights()
                    print('model weights backup')

            # backward AFTER model saved
            loss.backward()
            self.optimizer.step()
            t1 = time.time()

            ##########[ recording ]###########
            self._record(dur=t1 - t0,
                         train_loss=loss.item(),
                         train_acc=train_acc,
                         AMI=ami,
                         )

            dur_avg = np.average(self.dur)
            logfmt = "Epoch {:05d} | Train Acc: {:.4f} | AMI={:.4f} | Time: {:.4f}"
            self._cur_log = logfmt.format(
                self._cur_epoch,
                train_acc,
                ami,
                dur_avg)

            print(self._cur_log)
        self._cur_epoch_adopted = self._cur_epoch

    def eval_current(self,
                     feat_dict=None,
                     g=None,
                     **other_inputs):
        """ get the current states of the model output
        """
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
