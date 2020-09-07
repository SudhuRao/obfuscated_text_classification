# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:57:02 2020

@author: sudhanva
"""

from easydict import EasyDict as edict


__C                                 = edict()

#Get config by : from config import cfg
cfg                                 = __C



#Training parameters

__C.max_len                         = 500
__C.max_words                       = 500
__C.max_vocab                       = 27
__C.epochs                          = 100
__C.batch_size                      = 128
