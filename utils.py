#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import numpy as np
import gensim

def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = range(length)
    for j in xrange(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in xrange(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]


def load_word_id_mapping(word_id_file, encoding='utf8'):
    """
    :param word_id_file: word-id mapping file path
    :param encoding: file's encoding, for changing to unicode
    :return: word-id mapping, like hello=5
    """
    word_to_id = dict()
    for line in open(word_id_file):
        line = line.decode(encoding, 'ignore').lower().split()
        word_to_id[line[0]] = int(line[1])
    print '\nload word-id mapping done!\n'
    return word_to_id

def load_word_embedding(word_id_file, w2v_file, embedding_dim):
    word_to_id = load_word_id_mapping(word_id_file)
    model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True, unicode_errors='ignore')
    vocab = model.vocab
    w2v = []
    cnt = 0
    for k in word_to_id.keys():
        if k in vocab:
            w2v.append(model[k])
            cnt+=1
        else:
            w2v.append(np.random.uniform(-0.01, 0.01, (embedding_dim,)))
    w2v = np.asarray(w2v, dtype=np.float32)
    print cnt
    print len(word_to_id), len(w2v)
    return word_to_id, w2v


def change_y_to_onehot(y):
    from collections import Counter
    print Counter(y)
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_inputs_twitter_at(input_file, word_id_file, sentence_len, type_='', encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print 'load word-to-id done!'

    x, y, sen_len = [], [], []
    aspect_words = []
    lines = open(input_file).readlines()
    for i in xrange(0, len(lines), 3):
        words = lines[i].decode(encoding).lower().split()
        ids = []
        for word in words:
            if word in word_to_id:
                ids.append(word_to_id[word])
        # ids = list(map(lambda word: word_to_id.get(word, 0), words))
        if len(ids)==0:
            continue
        sen_len.append(len(ids))
        x.append(ids + [0] * (sentence_len - len(ids)))
        aspect_words.append(lines[i + 1].split()[0])
        y.append(lines[i + 2].split()[0])
    y = change_y_to_onehot(y)
    aspect_words = change_y_to_onehot(aspect_words)
    aspect_words = aspect_words.astype(np.float32)
    aspect_words -= 0.5
    for item in x:
        if len(item) != sentence_len:
            print 'aaaaa=', len(item)
    print "max", max(sen_len)
    x = np.asarray(x, dtype=np.int32)
    return x, np.asarray(sen_len), np.asarray(aspect_words), np.asarray(y)

def load_inputs_pediction(input_file, word_id_file, sentence_len, type_='', encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print 'load word-to-id done!'

    x, sen_len = [], []
    aspect_words = []
    lines = open(input_file).readlines()
    nums = []
    id = -1
    for i in xrange(0, len(lines), 2):
        words = lines[i].decode(encoding).lower().split()
        ids = []
        for word in words:
            if word in word_to_id:
                ids.append(word_to_id[word])
        id+=1
        # apppend twice because of two aspects
        if(len(ids)>sentence_len):
            continue
        nums.append(id)
        sen_len.append(len(ids))
        sen_len.append(len(ids))
        x.append(ids + [0] * (sentence_len - len(ids)))
        x.append(ids + [0] * (sentence_len - len(ids)))
        aspect_words.append("plot")
        aspect_words.append("cast")
    aspect_words = change_y_to_onehot(aspect_words)
    aspect_words = aspect_words.astype(np.float32)
    aspect_words -= 0.5
    for item in x:
        if len(item) != sentence_len:
            print 'aaaaa=', len(item)
    print "max", max(sen_len)
    x = np.asarray(x, dtype=np.int32)
    return x, np.asarray(sen_len), np.asarray(aspect_words),nums

