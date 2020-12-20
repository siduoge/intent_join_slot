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
class bilstm_crf(object):
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

    def bilstm_crf(self, scope):
        cell_fw = tf.contrib.rnn.LSTMCell(self.config.bilstm_hidden_dim)
        cell_bw = tf.contrib.rnn.LSTMCell(self.config.bilstm_hidden_dim)
        (output_fw_seq, output_bw_seq), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, 
                                                                             cell_bw=cell_bw, 
                                                                             inputs=self.query_embed,
                                                                             sequence_length=self.seq_len, 
                                                                             dtype=tf.float32)
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)

        output = tf.nn.dropout(output, self.keep_prob)
        
        slot_output = tf.layers.dense(output, self.config.slot_tag_size, activation=tf.nn.relu)
        self.slot_prob = slot_output
        memory_fw, hidden_fw = fw_state
        memory_bw, hidden_bw = bw_state
        intent_pb = tf.concat([hidden_fw, hidden_bw, memory_fw, memory_bw], axis=1)
        intent_pb = tf.nn.dropout(intent_pb, self.keep_prob)
        intent_output = tf.layers.dense(intent_pb, self.config.intent_classes, activation=tf.nn.relu)
        self.intent_prob = intent_output
        self.slot_score = fw_state
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
    def model(self):
        scope = "bilstm_crf"
        self.bilstm_crf(scope)
        self.seq_loss()




