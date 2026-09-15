# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 21:59:44 2020

@author: Xingyan Liu
"""

import dgl.function as _dgl_fn

# Compatibility shim for DGL >= 1.0: the message-function helpers
# ``dgl.function.copy_src`` and ``copy_dst`` were removed and renamed
# to ``copy_u`` and ``copy_v``. Restore the legacy names so that the
# model code (e.g. ``base_layers.py``) works on both old and new DGL.
if not hasattr(_dgl_fn, 'copy_src'):
    _dgl_fn.copy_src = _dgl_fn.copy_u
if not hasattr(_dgl_fn, 'copy_dst'):
    _dgl_fn.copy_dst = _dgl_fn.copy_v

from ._utils import *
from ._predict import *
from .loss import *
from ._predict import *
from .loss import *
from .cggc import CGGCNet
from .cgc import CGCNet
