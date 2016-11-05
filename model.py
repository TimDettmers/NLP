import tensorflow as tf
import numpy as np


class Model(object):
    def __init__(self, embedding_size, text_parser):
        self.embedding_size = embedding_size
        self.tp = text_parser

    def prediction(self, pX):
        embeddings = tf.Variable(tf.random_uniform((len(self.tp.vocab),self.embedding_size), -1.0,1.0))
        h1 = tf.Variable(tf.truncated_normal((self.embedding_size*self.tp.sequence_length, len(self.tp.vocab))))
        b1 = tf.Variable(tf.zeros(len(self.tp.vocab)))


        X = tf.nn.embedding_lookup(embeddings, pX)
        print embeddings
        X = tf.reshape(X, [-1,self.embedding_size*self.tp.sequence_length])
        logits = tf.matmul(X, h1) + b1
        return logits

    def loss(self, logits, pY):
        loss_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, pY)
        mean_loss = tf.reduce_mean(loss_value)
        return mean_loss


    def optimize(self, loss, learning_rate):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimize_step = opt.minimize(loss)
        return optimize_step



    def evaluation(self, logits, pY):
        pY = tf.to_int32(pY)
        correct = tf.nn.in_top_k(logits, pY, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def generate(self, logits):
        char = tf.argmax(logits,1)
        return char
