# -*- coding: UTF-8 -*-
"""
@CreateDate: 2021/06/22
@Author: Xingyan Liu
@File: test_predictor.py
@Project: CAME
"""
import came
import logging


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')

    came.model._predict.__test__()
