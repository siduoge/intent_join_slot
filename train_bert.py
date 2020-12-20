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
from bert import optimization
from bert import modeling
from model.cnn_crf import cnn_crf
from model.bilstm_crf import bilstm_crf
from model.bilstm_base import bilstm_base
from model.bilstm_atten import bilstm_atten
from model.bilstm_bert import bilstm_bert
from util.data_util import *
from util.config import *
import datetime
import sys

def train_step(data_batch, slot_label_batch, intent_label_batch, seq_len_batch, input_mask_batch, segments_batch, sess, train_op, model):
    #print(data_batch)
    fedd_dict = {
            model.input_ids : data_batch,
            model.intent_label : intent_label_batch,
            model.slots_label : slot_label_batch,
            model.seq_len : seq_len_batch,
            model.input_mask : input_mask_batch,
            model.segment_ids : segments_batch,
            model.keep_prob : FLAGS.keep_prob}
    #print(fedd_dict)
    _, step, loss, intent_acc, slot_score, slot_path = sess.run([train_op, global_step, model.loss, model.intent_accuracy, model.slot_score, model.slot_tags], fedd_dict)
    slot_label_batch = np.array(slot_label_batch)
    slot_f1, slot_pre, slot_recall = id2slot_label(slot_path, slot_label_batch, dic_slot)
    time_str = datetime.datetime.now().isoformat()
    #print("{}: step {}, loss: {:g}, intent_acc: {:g}, slot_f1: {:g}, slot_pre: {:g}, slot_recall: {:g},".format(time_str, step, loss, intent_acc, slot_f1, slot_pre, slot_recall))

def dev_step(data_batch, slot_label_batch, intent_label_batch, seq_len_batch, dev_input_mask, dev_segments, sess,  model):
    feed_dict = {
            model.input_ids : data_batch,
            model.intent_label : intent_label_batch,
            model.slots_label : slot_label_batch,
            model.seq_len : seq_len_batch,
            model.input_mask : dev_input_mask,
            model.segment_ids : dev_segments,
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

dev_data, dev_slot_label_data, dev_intent_labels, dev_seq_len, dev_input_mask, dev_segments = load_data_bert(intent_dic, slot_dic, vocab_dic, FLAGS.test_intent, FLAGS.test_slot, FLAGS.test_data, FLAGS.max_len)

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
            model = bilstm_base(FLAGS)
        elif FLAGS.model == "bilstm_atten":
            model = bilstm_atten(FLAGS)
        elif FLAGS.model == "bilstm_bert":
            model = bilstm_bert(FLAGS)
        #global_step = tf.Variable(0, name="global_step", trainable=False)
        global_step = tf.train.get_or_create_global_step()
        num_train_steps = 10000
        learning_rate = tf.constant(value=FLAGS.lr, shape=[], dtype=tf.float32)
        num_warmup_steps = int(num_train_steps * 0.1)
        learning_rate = tf.train.polynomial_decay(FLAGS.lr,global_step,num_train_steps,end_learning_rate=0.0,power=1.0,cycle=False)
        if num_warmup_steps:
            global_steps_int = tf.cast(global_step, tf.int32)
            warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = FLAGS.lr * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
        optimizer = optimization.AdamWeightDecayOptimizer(learning_rate=learning_rate,
                                                          weight_decay_rate=0.01,
                                                          beta_1=0.9,
                                                          beta_2=0.999,
                                                          epsilon=1e-6,
                                                          exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
        #saver = tf.train.Saver(tf.global_variables())
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, trainable_variables), clip_norm=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables),global_step=global_step)
        new_global_step = global_step + 1
        train_op = tf.group(train_op, [global_step.assign(new_global_step)])
        #bert模型参数初始化的地方
        #init_checkpoint = "chinese_L-12_H-768_A-12/bert_model.ckpt"
        use_tpu = False
        # 获取模型中所有的训练参数。
        tvars = tf.trainable_variables()
        # 加载BERT模型
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                       FLAGS.init_checkpoint)
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        sess.run(tf.global_variables_initializer())
        train_data, slot_label_data, intent_labels, seq_len, train_input_mask, train_segments = load_data_bert(intent_dic, slot_dic, vocab_dic, FLAGS.train_intent, FLAGS.train_slot, FLAGS.train_data, FLAGS.max_len)
        """
        print(train_data[0])
        print(slot_label_data[0])
        print(intent_labels[0])
        print(seq_len[0])
        print(train_input_mask[0])
        print(train_segments[0])
        """
        #sys.exit()
        batches = batch_iter(list(zip(train_data, slot_label_data, intent_labels, seq_len, train_input_mask, train_segments)), FLAGS.batch_size, FLAGS.num_epochs, True)
        iter_n = 0
        
        for batch in batches:
            iter_n += 1
            data_batch, slot_label_batch, intent_label_batch, seq_len_batch, input_mask_batch, segments_batch = zip(*batch)
            """
            print(data_batch[0])
            print(slot_label_batch[0])
            print(intent_label_batch[0])
            print(seq_len_batch[0])
            print(input_mask_batch[0])
            print(segments_batch[0]) 
            """
            train_step(data_batch, slot_label_batch, intent_label_batch, seq_len_batch, input_mask_batch, segments_batch, sess, train_op, model)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation {}:".format(iter_n))
                dev_step(dev_data, dev_slot_label_data, dev_intent_labels, dev_seq_len, dev_input_mask, dev_segments, sess, model)
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))








