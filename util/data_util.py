#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: data_util.py
Author: DST(DST@baidu.com)
Date: 2020/11/24 15:02:20
"""
import sys
import numpy as np
from bert import tokenization
from util.config import *

tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.bert_vocab)

def load_data_bert(intent_dic, slot_dic, vocab_dic, intent_path, slot_path, data_path, max_len):
    fr_data = open(data_path, "r")
    fr_intent = open(intent_path, "r")
    fr_slot = open(slot_path, "r")
    intent_list = fr_intent.readlines()
    slot_list = fr_slot.readlines()
    data_list = fr_data.readlines()
    fr_data.close()
    fr_intent.close()
    fr_slot.close()

    intent_labels = []
    slot_label_data = []
    train_data = []
    seq_len = []
    segments = []
    input_masks = []
    for i, cur_data in enumerate(data_list):
        slot_label = []
        input_mask = []
        data_ids  = []
        cur_slot = slot_list[i].strip()
        cur_intent = intent_list[i]

        tokens_data = cur_data.split()
        tokens_slot = cur_slot.split()
        #print(tokens_data)
        #print(tokens_slot)
        data_ids.append("[CLS]")
        slot_label.append(slot_dic["[CLS]"])
        input_mask.append(1)
        for i, val in enumerate(tokens_data):
            val = tokenizer.tokenize(val)
            data_ids.extend(val)
            for j, piece in enumerate(val):
                input_mask.append(1)
                if j == 0:
                    slot_label.append(slot_dic.get(tokens_slot[i], slot_dic["_UNK"]))
                else:
                    slot_label.append(slot_dic["[##WordPiece]"])
        data_ids = data_ids[:max_len-1]
        slot_label = slot_label[:max_len-1]
        input_mask = input_mask[:max_len-1]
        
        data_ids.append("[SEP]")
        slot_label.append(slot_dic["[SEP]"])
        input_mask.append(1)
        slot_len = min(max_len, len(slot_label))
        data = tokenizer.convert_tokens_to_ids(data_ids)
        if max_len > len(data):
            pad_data = [0 for i in range(max_len - len(data))]
            data.extend(pad_data)
            slot_label.extend(pad_data)
            input_mask.extend(pad_data)

        segment_ids = [0 for i in range(max_len)]
        intent_labels.append(intent_dic.get(cur_intent, intent_dic['_UNK']))
        seq_len.append(slot_len)
        slot_label_data.append(slot_label)
        segments.append(segment_ids)
        input_masks.append(input_mask)
        train_data.append(data)
        #break
    return np.array(train_data), np.array(slot_label_data), np.array(intent_labels), np.array(seq_len), np.array(input_masks), np.array(segments)

def createVocabulary(input_path, output_path, no_pad=False):
    vocab = {}
    with open(input_path, 'r') as fd, open(output_path, 'w+') as out:
        for line in fd:
            line = line.rstrip('\r\n')
            words = line.split()
            for w in words:
                if w == '_UNK':
                    break
                if str.isdigit(w) == True:
                    w = '0'
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        if no_pad == False:
            vocab = ['_PAD', '_UNK'] + sorted(vocab, key=vocab.get, reverse=True)
        else:
            vocab = ['_UNK'] + sorted(vocab, key=vocab.get, reverse=True)

        for v in vocab:
            out.write(v+'\n')

def loadVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = []
    rev = []
    with open(path) as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x,y) for (y,x) in enumerate(rev)])
        reverse_vocab = dict([(y,x) for (y,x) in enumerate(rev)])

    return vocab, reverse_vocab

def sentenceToIds(data, ret, vocab):
    if isinstance(data, str):
        words = data.split()
    elif isinstance(data, list):
        words = data
    else:
        raise TypeError('data should be a string or a list contains words')

    ids = []
    for i, w in enumerate(words):
        if str.isdigit(w) == True:
            w = '0'
        if i < len(ret):
            ret[i] = (vocab.get(w, vocab['_UNK']))

def load_data(intent_dic, slot_dic, vocab_dic, intent_path, slot_path, data_path, max_len):
    fr_data = open(data_path, "r")
    fr_intent = open(intent_path, "r")
    fr_slot = open(slot_path, "r")
    intent_list = fr_intent.readlines()
    slot_list = fr_slot.readlines()
    data_list = fr_data.readlines()
    fr_data.close()
    fr_intent.close()
    fr_slot.close()
    
    intent_labels = []
    slot_label_data = []
    train_data = []
    seq_len = []

    for i, cur_data in enumerate(data_list):
        slot_label = np.zeros(max_len)
        data = np.zeros(max_len)
        cur_slot = slot_list[i].strip()
        cur_intent = intent_list[i].strip().split("#")[0]

        sentenceToIds(cur_data, data, vocab_dic)
        sentenceToIds(cur_slot, slot_label, slot_dic)
        intent_labels.append(intent_dic.get(cur_intent, intent_dic['_UNK']))
        seq_len.append(min(len(cur_data.split()), max_len))
        slot_label_data.append(slot_label)
        #print(data)
        train_data.append(data) 
    return np.array(train_data), np.array(slot_label_data), np.array(intent_labels), np.array(seq_len)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    #shuffle = False
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def id2slot_label(pred_slot_id, slot_label_id, slot_dic):
        pred_slot_id_list = pred_slot_id.tolist()
        slot_label_id_list = slot_label_id.tolist()
        f1_arr = []
        pre_arr = []
        recall_arr = []
        pred_slots = []
        slot_labels = []
        for i in range(len(pred_slot_id_list)):
            pred_slot_sentence = pred_slot_id_list[i]
            slot_label_sentence = slot_label_id_list[i]
            pred_slot = []
            slot_label_id = []
            for i, val in enumerate(slot_label_sentence):
                if int(val) == 0:
                    break
                """
                if int(val) == 3:
                    continue
                if int(val) == 1:
                    continue
                if int(val) == 2:
                    continue
                """
                slot_label = slot_dic[val]
                pred_label = slot_dic[pred_slot_sentence[i]]
                pred_slot.append(pred_label)
                slot_label_id.append(slot_label)
            pred_slots.append(pred_slot[:])
            slot_labels.append(slot_label_id[:])
            #print(pred_slot)
            #print(slot_label_id)
            #print("------------")
        f1, precision, recall = computeF1Score(slot_labels, pred_slots)
        return f1, precision, recall
        #f1_arr.append(f1)
        #    pre_arr.append(precision)
        #    recall_arr.append(recall)
        #return sum(f1_arr)/len(f1_arr), sum(pre_arr)/len(pre_arr), sum(recall_arr)/len(recall_arr)

# compute f1 score is modified from conlleval.pl
def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart = False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if prevTag == 'E' and tag == 'E':
        chunkStart = True
    if prevTag == 'E' and tag == 'I':
        chunkStart = True
    if prevTag == 'O' and tag == 'E':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart

def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd = False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag == 'E' and tag == 'E':
        chunkEnd = True
    if prevTag == 'E' and tag == 'I':
        chunkEnd = True
    if prevTag == 'E' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd

def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType

def computeF1Score(correct_slots, pred_slots):
    correctChunk = {}
    correctChunkCnt = 0
    foundCorrect = {}
    foundCorrectCnt = 0
    foundPred = {}
    foundPredCnt = 0
    correctTags = 0
    tokenCount = 0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        inCorrect = False
        lastCorrectTag = 'O'
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if inCorrect == True:
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                   __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                   (lastCorrectType == lastPredType):
                    inCorrect = False
                    correctChunkCnt += 1
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1
                    else:
                        correctChunk[lastCorrectType] = 1
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                     __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                     (correctType != predType):
                    inCorrect = False

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
               __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
               (correctType == predType):
                inCorrect = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1
                else:
                    foundCorrect[correctType] = 1

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                foundPredCnt += 1
                if predType in foundPred:
                    foundPred[predType] += 1
                else:
                    foundPred[predType] = 1

            if correctTag == predTag and correctType == predType:
                correctTags += 1

            tokenCount += 1

            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if inCorrect == True:
            correctChunkCnt += 1
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1
            else:
                correctChunk[lastCorrectType] = 1

    if foundPredCnt > 0:
        precision = 100*correctChunkCnt/foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 100*correctChunkCnt/foundCorrectCnt
    else:
        recall = 0

    if (precision+recall) > 0:
        f1 = (2*precision*recall)/(precision+recall)
    else:
        f1 = 0

    return f1, precision, recall
