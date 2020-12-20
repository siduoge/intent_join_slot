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
from util.config import *
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
class cnn_crf(object):
    def __init__(self, FLAGS):
        self.config = FLAGS
        self.add_placeholder()
        self.input2embed()
        self.model()

    def add_placeholder(self):
        self.input_query = tf.placeholder(tf.int32, [None, self.config.max_len], name='query_input')
        self.seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
        self.intent_label = tf.placeholder(tf.int32, [None], name="intent_label")
        self.slots_label = tf.placeholder(tf.int32, [None, self.config.max_len], name='slot_label')
        self.keep_prob = tf.placeholder(tf.float32, name='slot_label')
        self.slot_tag_size = self.config.slot_tag_size

    def input2embed(self):
        self.query_embedding = tf.get_variable(name='word_embedding',
                                                   shape=[self.config.word_vocab_size, self.config.word_embedding_dim],
                                                   dtype=tf.float32,
                                                   initializer=tf.random_uniform_initializer(maxval=-1., minval=1., seed=0))
        self.query_embed = tf.nn.embedding_lookup(self.query_embedding, self.input_query)
        # column -> row
        self.transition = tf.get_variable(name='slot_transition', 
                                         shape=[self.slot_tag_size, self.slot_tag_size], 
                                         dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(maxval=-1., minval=1., seed=0))

    def cnn_crf(self, scope):
        with tf.name_scope(scope):
            conv_output = tf.layers.conv1d(self.query_embed, self.config.conv_filters, self.config.conv_kernel_size, self.config.conv_strides, padding="SAME", activation=tf.nn.relu)
            # max pool = 3 []
            conv_output= tf.nn.dropout(conv_output, self.keep_prob)
            slot_pool_output = tf.layers.max_pooling1d(conv_output, self.config.pool_size, self.config.pool_strides, padding="SAME")
            slot_pool_output = tf.nn.dropout(slot_pool_output, self.keep_prob)
            slot_output = tf.layers.dense(slot_pool_output, self.config.slot_tag_size, activation=tf.nn.relu)
            self.slot_prob = slot_output 

            intent_output = tf.layers.dense(tf.reduce_sum(conv_output, axis=1), self.config.intent_classes, activation=tf.nn.relu) #[b, hidden]
            self.intent_prob = intent_output 
            self.slot_tags, self.slot_score = tf.contrib.crf.crf_decode(self.slot_prob, self.transition, self.seq_len)



    def seq_loss(self):
        with tf.name_scope("loss"):
            slot_loss, self.transition = tf.contrib.crf.crf_log_likelihood(self.slot_prob,
                                                      self.slots_label,
                                                      self.seq_len,
                                                      transition_params=self.transition)
            
            intent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.intent_label, logits=self.intent_prob)
            self.loss = tf.reduce_mean(intent_loss) + 0.5 * tf.reduce_mean(-slot_loss)
        
        with tf.name_scope("accuarcy"):
            intent_predict = tf.cast(tf.argmax(self.intent_prob, axis=-1, name="predict"), dtype=tf.int32)
            correct_predictions = tf.equal(intent_predict, self.intent_label, name='correct_predictions')
            self.intent_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.slot_tags = tf.cast(tf.argmax(self.slot_prob, axis=-1, name="predict"), dtype=tf.int32)
    def model(self):
        scope = "cnn_crf"
        self.cnn_crf(scope)
        #self.bilstm_crf(scope)
        self.seq_loss()




