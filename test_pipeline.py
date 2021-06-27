# -*- coding: utf-8 -*-
"""
Created on Wed May  5 20:08:52 2021

@author: Xingyan Liu
"""
import CAME
from CAME import pipeline_supervised
import matplotlib as mpl
import logging


try:
    mpl.use('agg')
except Exception as e:
    print(f"An error occurred when setting matplotlib backend ({e})")

if __name__ == '__main__':


    # CAME.__test1__()
    CAME.__test2__(100, batch_size=1024)
    # pipeline_supervised.__test2_sup__(5)

