# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 21:59:44 2020

@author: Xingyan Liu
"""

from .utils import base
from .utils.base import (
        save_pickle,
        load_pickle,
        save_json_dict,
        load_json_dict,
        check_dirs,
        write_info,
        make_nowtime_tag,
        subsample_each_group,
        )
from .utils import preprocess as pp
from .utils import plot as pl
from .utils.train import prepare4train, Trainer, SUBDIR_MODEL
from .utils.evaluation import (
        as_probabilities, 
        predict_from_logits, 
        predict, 
        accuracy
        )                       
from .utils.analyze import (
        load_dpair_and_model,
        weight_linked_vars, 
        make_abstracted_graph,
        )             

from .datapair.unaligned import datapair_from_adatas, DataPair
from .datapair.aligned import aligned_datapair_from_adatas, AlignedDataPair
from .model.cggc import CGGCNet
from .model.cgc import CGCNet
from .PARAMETERS import get_model_params, get_loss_params
from .pipeline import __test1__, __test2__


__version__ = "0.0.1"