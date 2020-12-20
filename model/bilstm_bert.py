#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: cnn_crf.py
Author: DST(DST@baidu.com)
Date: 2020/11/24 16:37:30
"""
import tensorflow as tf
from model.model_base import *
from bert import modeling
from util.config import *
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
class bilstm_bert(object):
    def __init__(self, FLAGS):
        self.config = FLAGS
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_json)
        self.bert_config = bert_config
        self.add_placeholder()
        self.input2embed()
        self.model()

    def add_placeholder(self):
        self.input_ids = tf.placeholder(tf.int32, [None, self.config.max_len], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [None, self.config.max_len], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [None, self.config.max_len], name='segment_ids')
        self.seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
        self.intent_label = tf.placeholder(tf.int32, [None], name="intent_label")
        self.slots_label = tf.placeholder(tf.int32, [None, self.config.max_len], name='slot_label')
        self.keep_prob = tf.placeholder(tf.float32, name='slot_label')
        self.slot_tag_size = self.config.slot_tag_size

    def input2embed(self):
        self.transition = tf.get_variable(name='slot_transition',
                                         shape=[self.slot_tag_size, self.slot_tag_size],
                                         dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(maxval=-1., minval=1., seed=0))
    
    def bert_base(self, scope):
        self.bert_model = modeling.BertModel(config=self.bert_config,
                                   is_training=True,
                                   input_ids=self.input_ids,
                                   input_mask=self.input_mask,
                                   token_type_ids=self.segment_ids,
                                   use_one_hot_embeddings=False)
        output_layer = self.bert_model.get_sequence_output()
        sen_layer = self.bert_model.get_pooled_output()
        self.slot_prob = tf.layers.dense(output_layer, self.config.slot_tag_size, activation=tf.nn.relu)
        self.intent_prob = tf.layers.dense(sen_layer, self.config.intent_classes, activation=tf.nn.relu)
        self.slot_tags, self.slot_score = tf.contrib.crf.crf_decode(self.slot_prob, self.transition, self.seq_len)
    def seq_loss(self):
        with tf.name_scope("loss"):
            slot_loss, self.transition = tf.contrib.crf.crf_log_likelihood(self.slot_prob,
                                                      self.slots_label,
                                                      self.seq_len,
                                                      transition_params=self.transition)

            intent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.intent_label, logits=self.intent_prob)
            self.loss = tf.reduce_mean(intent_loss) + tf.reduce_mean(-slot_loss)
        with tf.name_scope("accuarcy"):
            intent_predict = tf.cast(tf.argmax(self.intent_prob, axis=-1, name="predict"), dtype=tf.int32)
            correct_predictions = tf.equal(intent_predict, self.intent_label, name='correct_predictions')
            self.intent_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    def model(self):
        scope = "bert_base"
        self.bert_base(scope)
        self.seq_loss()




