#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: test_data.py
Author: DST(DST@baidu.com)
Date: 2020/12/02 17:15:21
"""

import tensorflow as tf
import numpy as np
import os
import time
from model.cnn_crf import cnn_crf
from model.bilstm_crf import bilstm_crf
from model.bilstm_base import bilstm_base
from model.bilstm_atten import bilstm_atten
from util.data_util import *
from util.config import *
import datetime
import sys

slot2index, index2slot = loadVocabulary(FLAGS.slot_vocab)

slot_label = np.random.randint(1, 122, (2, 20))
pred_slot = np.random.randint(1, 122, (2, 20))
#print(slot_label)
#print(pred_slot)
#print(id2slot_label(pred_slot, slot_label, index2slot))

intent_dic, _1 = loadVocabulary(FLAGS.intent_vocab)
slot_dic, dic_slot = loadVocabulary(FLAGS.slot_vocab)
vocab_dic, _2  = loadVocabulary(FLAGS.word_vocab)

data, slot_label, intent_label, seq_len, input_masks, segments  = load_data_bert(intent_dic, slot_dic, vocab_dic, FLAGS.test_intent, FLAGS.test_slot, FLAGS.test_data, FLAGS.max_len)
print("data: ", data[0])
print("slot: ", slot_label[0])
print("intent: ", intent_label[0])
print("seq_len: ", seq_len[0])
print("input_mask: ", input_masks[0])
print("segment: ", segments[0])
