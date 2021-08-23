# -*- coding: utf-8 -*-
"""
Created on Wed May  5 20:08:52 2021

@author: Xingyan Liu
"""
import came
from came import pipeline_supervised
import matplotlib as mpl
import logging


try:
    mpl.use('agg')
except Exception as e:
    print(f"An error occurred when setting matplotlib backend ({e})")

if __name__ == '__main__':

    came.__test1__(6, batch_size=2048)
    came.__test2__(6, batch_size=None)
    # pipeline_supervised.__test2_sup__(5)

