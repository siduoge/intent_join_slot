#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: train.py
Author: DST(DST@baidu.com)
Date: 2020/12/01 16:44:54
"""
import tensorflow as tf
import numpy as np
import os
import time
from model.cnn_crf import cnn_crf
from model.bilstm_crf import bilstm_crf
from model.bilstm_base import bilstm_base
from model.bilstm_atten import bilstm_atten
from model.bilstm_slot_gated import bilstm_slot_gated
from util.data_util import *
from util.config import *
import datetime
import sys

def train_step(data_batch, slot_label_batch, intent_label_batch, seq_len_batch, sess, train_op, model):
    #print(data_batch)
    fedd_dict = {
            model.input_query : data_batch,
            model.intent_label : intent_label_batch,
            model.slots_label : slot_label_batch,
            model.seq_len : seq_len_batch,
            model.keep_prob : FLAGS.keep_prob}
    #print(fedd_dict)
    _, step, loss, intent_acc, slot_score, slot_path = sess.run([train_op, global_step, model.loss, model.intent_accuracy, model.slot_score, model.slot_tags], fedd_dict)
    slot_label_batch = np.array(slot_label_batch)
    #slot_f1, slot_pre, slot_recall = id2slot_label(slot_path, slot_label_batch, dic_slot)
    time_str = datetime.datetime.now().isoformat()
    #print("{}: step {}, loss: {:g}, intent_acc: {:g}, slot_f1: {:g}, slot_pre: {:g}, slot_recall: {:g},".format(time_str, step, loss, intent_acc, slot_f1, slot_pre, slot_recall))

def dev_step(data_batch, slot_label_batch, intent_label_batch, seq_len_batch, sess,  model):
    feed_dict = {
            model.input_query : data_batch,
            model.intent_label : intent_label_batch,
            model.slots_label : slot_label_batch,
            model.seq_len : seq_len_batch,
            model.keep_prob : 1.0}
    #print(feed_dict)
    step, loss, intent_acc, slot_score, slot_path = sess.run([global_step, model.loss, model.intent_accuracy, model.slot_score, model.slot_tags], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    slot_label_batch = np.array(slot_label_batch)
    slot_f1, slot_pre, slot_recall = id2slot_label(slot_path, slot_label_batch, dic_slot)
    print("{}: step {}, loss: {:g}, intent_acc: {:g},slot_f1: {:g}, slot_pre: {:g}, slot_recall: {:g},".format(time_str, step, loss, intent_acc, slot_f1, slot_pre, slot_recall))
    


intent_dic, _1 = loadVocabulary(FLAGS.intent_vocab)
slot_dic, dic_slot = loadVocabulary(FLAGS.slot_vocab)
vocab_dic, _2  = loadVocabulary(FLAGS.word_vocab)

dev_data, dev_slot_label_data, dev_intent_labels, dev_seq_len = load_data(intent_dic, slot_dic, vocab_dic, FLAGS.test_intent, FLAGS.test_slot, FLAGS.test_data, FLAGS.max_len)

os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(FLAGS.gpu_id)
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        if FLAGS.model == "cnn_crf":
            model = cnn_crf(FLAGS)
        elif FLAGS.model == "bilstm_crf":
            model = bilstm_crf(FLAGS)
        elif FLAGS.model == "bilstm_base":
            print("wwefwefewfe1")
            model = bilstm_base(FLAGS)
        elif FLAGS.model == "bilstm_atten":
            model = bilstm_atten(FLAGS)
        elif FLAGS.model == "bilstm_slot_gated":
            print("1efrvervgerv")
            model = bilstm_slot_gated(FLAGS)
        saver = tf.train.Saver(tf.global_variables())
        global_step = tf.Variable(0, name="global_step", trainable=False)
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, trainable_variables), 10)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables),global_step=global_step)
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        sess.run(tf.global_variables_initializer())
        train_data, slot_label_data, intent_labels, seq_len = load_data(intent_dic, slot_dic, vocab_dic, FLAGS.train_intent, FLAGS.train_slot, FLAGS.train_data, FLAGS.max_len)
        #print(train_data)
        #sys.exit()
        batches = batch_iter(list(zip(train_data, slot_label_data, intent_labels, seq_len)), FLAGS.batch_size, FLAGS.num_epochs, True)
        iter_n = 0
        
        for batch in batches:
            iter_n += 1
            data_batch, slot_label_batch, intent_label_batch, seq_len_batch = zip(*batch)
            #print(data_batch)
            train_step(data_batch, slot_label_batch, intent_label_batch, seq_len_batch, sess, train_op, model)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation {}:".format(iter_n))
                dev_step(dev_data, dev_slot_label_data, dev_intent_labels, dev_seq_len, sess, model)
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))








