#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com

import os
import tensorflow as tf
from utils import batch_index, load_word_embedding, load_inputs_twitter_at, load_inputs_pediction


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 50, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 200, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('n_aspect', 2, 'number of distinct aspect class')
tf.app.flags.DEFINE_integer('max_sentence_len', 75, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 50, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob1', 1.0, 'dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'dropout keep prob')


tf.app.flags.DEFINE_string('train_file_path', 'data/douban_train', 'training file')
tf.app.flags.DEFINE_string('test_file_path', 'data/douban_test', 'testing file')
tf.app.flags.DEFINE_string('predict_file_path', 'data/predict/Thor: Ragnarok1', 'testing file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/cn.skipgram.bin', 'embedding file')
tf.app.flags.DEFINE_string('word_id_file_path', 'data/word_id', 'word-id mapping file')
tf.app.flags.DEFINE_string('method', 'AT', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('t', 'last', 'model type: ')
tf.app.flags.DEFINE_string('mode', 'predict', 'predict or train')

class LSTM(object):

    def __init__(self, embedding_dim=100, batch_size=64, n_hidden=100, learning_rate=0.01,n_aspect=2,
                 n_class=3, max_sentence_len=50, l2_reg=0., display_step=4, n_iter=100, type_=''):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_aspect = n_aspect
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.display_step = display_step
        self.n_iter = n_iter
        self.type_ = type_
        self.word_id_mapping, self.w2v = load_word_embedding(FLAGS.word_id_file_path, FLAGS.embedding_file_path, self.embedding_dim)
        self.word_embedding = tf.Variable(self.w2v, dtype=tf.float32, name='word_embedding')

        self.keep_prob1 = tf.placeholder(tf.float32)
        self.keep_prob2 = tf.placeholder(tf.float32)
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x')
            self.y = tf.placeholder(tf.int32, [None, self.n_class], name='y')
            self.sen_len = tf.placeholder(tf.int32, None, name='sen_len')
            self.aspect = tf.placeholder(tf.float32, [None,self.n_aspect], name='aspect_one_hot')#cast and plot

        with tf.name_scope('weights'):
            self.weights = {
                'softmax': tf.get_variable(
                    name='softmax_w',
                    shape=[self.n_hidden, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax': tf.get_variable(
                    name='softmax_b',
                    shape=[self.n_class],
                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        self.W = tf.get_variable(
            name='W',
            shape=[self.n_hidden + self.n_aspect, self.n_hidden + self.n_aspect],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.w = tf.get_variable(
            name='w',
            shape=[self.n_hidden + self.n_aspect, 1],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wp = tf.get_variable(
            name='Wp',
            shape=[self.n_hidden, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wx = tf.get_variable(
            name='Wx',
            shape=[self.n_hidden, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )

    def dynamic_rnn(self, cell, inputs, length, max_len, scope_name, out_type='all'):
        outputs, state = tf.nn.dynamic_rnn(
            cell(self.n_hidden),
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope_name
        )  # outputs -> batch_size * max_len * n_hidden
        batch_size = tf.shape(outputs)[0]
        if out_type == 'last':
            index = tf.range(0, batch_size) * max_len + (length - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, self.n_hidden]), index)  # batch_size * n_hidden
        elif out_type == 'all_avg':
            outputs = LSTM.reduce_mean(outputs, length)
        return outputs

    def AT(self, inputs, target, type_=''):
        print('I am AT.')
        batch_size = tf.shape(inputs)[0]
        target = tf.reshape(target, [-1, 1, self.n_aspect])#cast,plot
        #target = tf.cast(target, tf.float32)
        target = tf.ones([batch_size, self.max_sentence_len, self.n_aspect], dtype=tf.float32) * target
        in_t = tf.concat([inputs, target], self.n_aspect)
        in_t = tf.nn.dropout(in_t, keep_prob=self.keep_prob1)
        cell = tf.nn.rnn_cell.LSTMCell
        hiddens = self.dynamic_rnn(cell, in_t, self.sen_len, self.max_sentence_len, 'AT', 'all')

        h_t = tf.reshape(tf.concat([hiddens, target], 2), [-1, self.n_hidden + self.n_aspect])

        M = tf.matmul(tf.tanh(tf.matmul(h_t, self.W)), self.w)
        alpha = LSTM.softmax(tf.reshape(M, [-1, 1, self.max_sentence_len]), self.sen_len, self.max_sentence_len)
        self.alpha = tf.reshape(alpha, [-1, self.max_sentence_len])

        r = tf.reshape(tf.matmul(alpha, hiddens), [-1, self.n_hidden])
        index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len - 1)
        hn = tf.gather(tf.reshape(hiddens, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        h = tf.tanh(tf.matmul(r, self.Wp) + tf.matmul(hn, self.Wx))

        return LSTM.softmax_layer(h, self.weights['softmax'], self.biases['softmax'], self.keep_prob2)

    @staticmethod
    def softmax_layer(inputs, weights, biases, keep_prob):
        with tf.name_scope('softmax'):
            outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
            predict = tf.matmul(outputs, weights) + biases
            predict = tf.nn.softmax(predict)
        return predict

    @staticmethod
    def reduce_mean(inputs, length):
        """
        :param inputs: 3-D tensor
        :param length: the length of dim [1]
        :return: 2-D tensor
        """
        length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
        inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
        return inputs

    @staticmethod
    def softmax(inputs, length, max_length):
        inputs = tf.cast(inputs, tf.float32)
        max_axis = tf.reduce_max(inputs, 2, keep_dims=True)
        inputs = tf.exp(inputs - max_axis)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
        return inputs / _sum

    def run(self):
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        aspect = self.aspect
        if FLAGS.method == 'AE':
            prob = self.AE(inputs, aspect, FLAGS.t)
        elif FLAGS.method == 'AT':
            prob = self.AT(inputs, aspect, FLAGS.t)

        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prob, self.y))
            cost = - tf.reduce_mean(tf.cast(self.y, tf.float32) * tf.log(prob)) + sum(reg_loss)

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=global_step)
            # optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            true_y = tf.argmax(self.y, 1)
            pred_y = tf.argmax(prob, 1)
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
            _acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
                FLAGS.keep_prob1,
                FLAGS.keep_prob2,
                FLAGS.batch_size,
                FLAGS.learning_rate,
                FLAGS.l2_reg,
                FLAGS.max_sentence_len,
                FLAGS.embedding_dim,
                FLAGS.n_hidden,
                FLAGS.n_class
            )
            summary_loss = tf.summary.scalar('loss' + title, cost)
            summary_acc = tf.summary.scalar('acc' + title, _acc)
            train_summary_op =  tf.summary.merge([summary_loss, summary_acc])
            validate_summary_op =  tf.summary.merge([summary_loss, summary_acc])
            test_summary_op =  tf.summary.merge([summary_loss, summary_acc])
            import time
            timestamp = str(int(time.time()))
            _dir = 'logs/' + str(timestamp) + '_' + title
            train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(_dir + '/test', sess.graph)
            validate_summary_writer = tf.summary.FileWriter(_dir + '/validate', sess.graph)

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

            init = tf.global_variables_initializer()
            sess.run(init)

            # saver.restore(sess, 'models/logs/1481529975__r0.005_b2000_l0.05self.softmax/-1072')

            save_dir = 'models/' + _dir + '/'
            import os
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            tr_x, tr_sen_len, tr_target_word, tr_y = load_inputs_twitter_at(
                FLAGS.train_file_path,
                self.word_id_mapping,
                self.max_sentence_len,
                self.type_
            )
            te_x, te_sen_len, te_target_word, te_y = load_inputs_twitter_at(
                FLAGS.test_file_path,
                self.word_id_mapping,
                self.max_sentence_len,
                self.type_
            )

            max_acc = 0.
            max_alpha = None
            max_ty, max_py = None, None
            for i in range(self.n_iter):
                acc_tr, loss_tr, cnt_tr = 0., 0., 0
                for train, num_tr in self.get_batch_data(tr_x, tr_sen_len, tr_y, tr_target_word, self.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                    _, step, summary,_acc_tr,_loss_tr = sess.run([optimizer, global_step, train_summary_op,
                                                                  accuracy,cost], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                    acc_tr += _acc_tr
                    loss_tr += _loss_tr * num_tr
                    cnt_tr += num_tr

                acc, loss, cnt = 0., 0., 0
                flag = True
                summary, step = None, None
                alpha = None
                ty, py = None, None
                for test, num in self.get_batch_data(te_x, te_sen_len, te_y, te_target_word, 500, 1.0, 1.0, False):
                    _loss, _acc, _summary, _step, alpha, ty, py = sess.run([cost, accuracy, validate_summary_op, global_step, self.alpha, true_y, pred_y],
                                                            feed_dict=test)
                    acc += _acc
                    loss += _loss * num
                    cnt += num
                    if flag:
                        summary = _summary
                        step = _step
                        flag = False
                        alpha = alpha
                        ty = ty
                        py = py
                print('all samples={}, correct prediction={}'.format(cnt, acc))
                test_summary_writer.add_summary(summary, step)
                saver.save(sess, save_dir, global_step=step)
                print('Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, loss / cnt, acc / cnt))
                #print('mini-batch train loss={:.6f}, train acc={:.6f}'.format(i, loss_tr / cnt_tr, acc_tr / cnt_tr))
                if acc / cnt > max_acc:
                    max_acc = acc / cnt
                    max_alpha = alpha
                    max_ty = ty
                    max_py = py

            print('Optimization Finished! Max acc={}'.format(max_acc))
            fp = open('weight', 'w')
            for y1, y2, ws in zip(max_ty, max_py, max_alpha):
                fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws]) + '\n')

            print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
                self.learning_rate,
                self.n_iter,
                self.batch_size,
                self.n_hidden,
                self.l2_reg
            ))

    def get_batch_data(self, x, sen_len, y, target_words, batch_size, keep_prob1, keep_prob2, is_shuffle=True):
        if y!=None:
            for index in batch_index(len(y), batch_size, 1, is_shuffle):
                feed_dict = {
                    self.x: x[index],
                    self.y: y[index],
                    self.sen_len: sen_len[index],
                    self.aspect: target_words[index],
                    self.keep_prob1: keep_prob1,
                    self.keep_prob2: keep_prob2,
                }
                yield feed_dict, len(index)
        else:
            for index in batch_index(len(sen_len), batch_size, 1, None):
                feed_dict = {
                    self.x: x[index],
                    self.sen_len: sen_len[index],
                    self.aspect: target_words[index],
                    self.keep_prob1: keep_prob1,
                    self.keep_prob2: keep_prob2,
                }
                yield feed_dict, len(index)

    def predict(self):
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        aspect = self.aspect
        if FLAGS.method == 'AE':
            prob = self.AE(inputs, aspect, FLAGS.t)
        elif FLAGS.method == 'AT':
            prob = self.AT(inputs, aspect, FLAGS.t)

        with tf.name_scope('predict'):
            result = tf.argmax(prob, 1)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("models/")
            ckpt_name = ckpt.model_checkpoint_path
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            saver.restore(sess, ckpt_name)
            print "Load sucess!"
            tr_x, tr_sen_len, tr_target_word,ids = load_inputs_pediction(
                FLAGS.predict_file_path,
                self.word_id_mapping,
                self.max_sentence_len,
                self.type_
            )
            results = []
            for train, _ in self.get_batch_data(tr_x, tr_sen_len, None, tr_target_word, self.batch_size,
                                                     FLAGS.keep_prob1, FLAGS.keep_prob2):
                res = sess.run(result, feed_dict=train)
                res = res.tolist()
                results += res
        #save the results
        predict_save_path = FLAGS.predict_file_path[:-1] + '2'
        with open(predict_save_path, 'w+') as f:
            for i in range(len(ids)):
                f.write(str(ids[i])+' ')
                f.write(str(results[2*i])+' ')
                f.write(str(results[2*i+1])+'\n')


def main(_):
    lstm = LSTM(
        embedding_dim=FLAGS.embedding_dim,
        batch_size=FLAGS.batch_size,
        n_hidden=FLAGS.n_hidden,
        learning_rate=FLAGS.learning_rate,
        n_aspect=FLAGS.n_aspect,
        n_class=FLAGS.n_class,
        max_sentence_len=FLAGS.max_sentence_len,
        l2_reg=FLAGS.l2_reg,
        display_step=FLAGS.display_step,
        n_iter=FLAGS.n_iter,
        type_=FLAGS.method
    )
    if(FLAGS.mode=="train"):
        lstm.run()
    else:
        lstm.predict()


if __name__ == '__main__':
    tf.app.run()
