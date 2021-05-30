# -*- coding: utf-8 -*-
"""
Created on Wed May  5 20:08:52 2021

@author: Xingyan Liu
"""
import CAME
from CAME import pipeline_supervised
import matplotlib as mpl

mpl.use('Agg')

if __name__ == '__main__':

    # CAME.__test1__()
    # CAME.__test2__()
    pipeline_supervised.__test2_sup__(100)
