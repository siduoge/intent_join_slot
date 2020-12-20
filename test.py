#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: test.py
Author: DST(DST@baidu.com)
Date: 2020/11/24 15:06:04
"""

import tensorflow as tf
import numpy as np
import os
from model.bilstm_slot_gated import bilstm_slot_gated

from util.config import *

batch_size = FLAGS.batch_size
batch_size = 1
max_len = FLAGS.max_len
word_vocab = FLAGS.word_vocab_size
intent_classes = FLAGS.intent_classes
slot_tag_size = FLAGS.slot_tag_size
input_query = np.random.randint(0, word_vocab, (batch_size, max_len))
seq_len = np.random.randint(0, max_len, (batch_size))
intent_label = np.random.randint(0, intent_classes, (batch_size))
slots_label = np.random.randint(0, slot_tag_size, (batch_size, max_len))
input_mask = np.ones((batch_size, max_len))
segment_ids = np.ones((batch_size, max_len))
os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(3)
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        model = bilstm_slot_gated(FLAGS)
        sess.run(tf.global_variables_initializer())
        feed_dict = {
                #model.input_ids: input_query,
                model.input_query: input_query,
                model.seq_len: seq_len,
                model.intent_label: intent_label,
                model.slots_label: slots_label,
                #model.input_mask: input_mask,
                #model.segment_ids: segment_ids,
                model.keep_prob: 0.5,
        }
        loss, intent_acc, slot_tags, slot_prob, intent_prob = sess.run([model.loss, model.intent_accuracy, model.slot_tags, model.slot_prob, model.intent_prob], feed_dict)
        print(loss)
        print(intent_acc)
        print(slot_tags)
        print(slot_prob)
        print(intent_prob)
