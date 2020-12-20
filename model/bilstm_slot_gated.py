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
class bilstm_slot_gated(object):
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
    
    def bilstm_slot_gated(self, scope):
        cell_fw = tf.contrib.rnn.LSTMCell(self.config.bilstm_hidden_dim)
        cell_bw = tf.contrib.rnn.LSTMCell(self.config.bilstm_hidden_dim)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob,
                                             output_keep_prob=self.keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob,
                                             output_keep_prob=self.keep_prob) 
        (output_fw_seq, output_bw_seq), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, 
                                                                             cell_bw=cell_bw, 
                                                                             inputs=self.query_embed,
                                                                             sequence_length=self.seq_len, 
                                                                             dtype=tf.float32)
        slot_inputs = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        orignal_hidden = tf.expand_dims(slot_inputs, 1) 
        memory_fw, hidden_fw = fw_state
        memory_bw, hidden_bw = bw_state
        intent_input = tf.concat([hidden_fw, hidden_bw, memory_fw, memory_bw], axis=1)
        
        with tf.variable_scope('slot_attn'):
            hidden_size = slot_inputs.get_shape()[2]   
            # use conv_2d to get wXh
            hidden_conv = tf.expand_dims(slot_inputs, 2) #[b, seq_len, 1, hidden]
            weight_1 = tf.get_variable("Attn_1", [1, 1, hidden_size, hidden_size])
            atten_features_1 = tf.nn.conv2d(hidden_conv, weight_1, [1, 1, 1, 1], "VALID")
            atten_features_1 = tf.reshape(atten_features_1, [-1, self.config.max_len, hidden_size])
            atten_features_1 = tf.expand_dims(atten_features_1, 1) #[b, 1, seq_len, hidden]
            # use layer_dense to get wXh
            wegiht_score = tf.get_variable("Attn_score", [hidden_size])
            atten_features_2 = tf.layers.dense(slot_inputs, hidden_size, activation=None)
            atten_features_2 = tf.expand_dims(atten_features_2, 2) #[b, seq_len, 1, hidden]
            #add broadcast
            atten_matrix = tf.reduce_sum(wegiht_score * tf.tanh(atten_features_1 + atten_features_2), axis=3) #[b, seq_len ,seq_len]
            atten_matrix = tf.nn.softmax(atten_matrix, axis=-1) 
            atten_matrix = tf.expand_dims(atten_matrix, -1) #[b, seq_len ,seq_len ,1]
            # dot broadcase
            atten_hidden = orignal_hidden * atten_matrix #[b, seq_len ,seq_len ,hidden]
            slot_context = tf.reduce_sum(atten_hidden, axis=2)
        with tf.variable_scope('intent_attn'):
            word_hidden = tf.expand_dims(slot_inputs, 2) #[b, seq_len, 1, hidden]
            weigths_1 = tf.get_variable("AttnW", [1, 1, hidden_size, hidden_size])
            atten_features = tf.nn.conv2d(word_hidden, weigths_1, [1, 1, 1, 1], "SAME") #[b, seq, 1, hidden]
            atten_features_2 = tf.layers.dense(intent_input, hidden_size, activation=None)
            atten_features_2 = tf.reshape(atten_features_2, [-1, 1, 1, hidden_size]) #[b, 1, 1, hidden]
            wegiht_intent_score = tf.get_variable("Atten_score", [hidden_size])
            intent_atten_matrix = tf.reduce_sum(wegiht_intent_score * tf.tanh(atten_features_1 + atten_features_2), [2,3]) #[b, seq_len]
            intent_atten_matrix = tf.nn.softmax(intent_atten_matrix, axis=-1)
            intent_atten_matrix = tf.expand_dims(intent_atten_matrix, -1)
            intent_atten_matrix = tf.expand_dims(intent_atten_matrix, -1) #[b, seq_len,1,1]
            d = tf.reduce_sum(intent_atten_matrix * word_hidden, [1, 2]) #[b, hidden]
            intent_output = tf.concat([d, intent_input], 1)
            intent_gate = tf.layers.dense(intent_output, hidden_size, activation=None)
        with tf.variable_scope('slot_gated'):
            intent_context = tf.reshape(intent_gate, [-1, 1, hidden_size]) #[b, 1, hidden]
            weight_gate = tf.get_variable("gate", [hidden_size])
            slot_gate = weight_gate * tf.tanh(slot_context + intent_context) #[b, seq_len, hidden]
            slot_gate = tf.expand_dims(tf.reduce_sum(slot_gate, [2]), -1)
            slot_gate = slot_context * slot_gate #[b, seq_len, hidden]
            slot_output = tf.concat([slot_gate, slot_inputs], axis=-1)
        self.intent_prob =  tf.layers.dense(intent_output, self.config.intent_classes, activation=tf.nn.relu)
        self.slot_prob = tf.layers.dense(slot_output, self.config.slot_tag_size, activation=tf.nn.relu)
        self.slot_score = weight_gate
    def seq_loss(self):
        with tf.name_scope("loss"):
            self.mask = tf.sequence_mask(self.seq_len, self.config.max_len, dtype=tf.float32, name='masks')
            #self.slot_prob = tf.Print(self.slot_prob, ["slot_prob", self.slot_prob[0]], summarize=100000)
            #self.slots_label = tf.Print(self.slots_label, ["self.slots_label", self.slots_label[0]], summarize=100000)
            slot_loss = tf.contrib.seq2seq.sequence_loss(self.slot_prob, self.slots_label, self.mask)
            #slot_loss = tf.Print(slot_loss, ["slot_loss", slot_loss], summarize=100)
            intent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.intent_label, logits=self.intent_prob)
            #intent_loss = tf.Print(intent_loss, ["intent_loss", intent_loss], summarize=100)
            self.loss = tf.reduce_mean(intent_loss) + tf.reduce_mean(slot_loss)
        with tf.name_scope("accuarcy"):
            intent_predict = tf.cast(tf.argmax(self.intent_prob, axis=-1, name="predict"), dtype=tf.int32)
            correct_predictions = tf.equal(intent_predict, self.intent_label, name='correct_predictions')
            self.intent_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.slot_tags = tf.cast(tf.argmax(self.slot_prob, axis=-1, name="predict"), dtype=tf.int32)
    def model(self):
        scope = "bilstm_base"
        self.bilstm_slot_gated(scope)
        self.seq_loss()




