import tensorflow as tf
import numpy as np


class Model(object):
    def __init__(self, embedding_size, text_parser):
        self.embedding_size = embedding_size
        self.tp = text_parser

    def prediction(self, pX):
        embeddings = tf.Variable(tf.random_uniform((len(self.tp.vocab),self.embedding_size), -1.0,1.0))
        filters=8
        k1 = tf.Variable(tf.truncated_normal((11,11,1,filters), stddev = 0.1))
        b1 = tf.Variable(tf.zeros((filters)))
        #h1 = tf.Variable(tf.truncated_normal((self.embedding_size*self.tp.sequence_length, len(self.tp.vocab))))
        #b1 = tf.Variable(tf.zeros(len(self.tp.vocab)))
        #self.h2 = tf.Variable(tf.truncated_normal((filters*self.embedding_size*self.tp.sequence_length, len(self.tp.vocab)),stddev=0.1))
        #b2 = tf.Variable(tf.zeros(len(self.tp.vocab)))

        #X = tf.reshape(X, [-1,self.embedding_size,self.tp.sequence_length,1])


        #X = tf.nn.embedding_lookup(embeddings, pX)
        #X = tf.reshape(X, [-1,self.embedding_size,self.tp.sequence_length,1])
        #conv1 = tf.nn.conv2d(X,k1,strides=[1,1,1,1], padding='SAME')
        #a1 = tf.nn.bias_add(conv1,b1)
        #y1 = tf.nn.relu(a1)

        #s = y1.get_shape().as_list()
        #y1Reshaped = tf.reshape(y1,[s[0], s[1]*s[2]*s[3]])


        #logits = tf.matmul(y1Reshaped, self.h2) + b2

        self.h1 = tf.Variable(tf.truncated_normal((self.embedding_size*self.tp.sequence_length, 1024)))
        b1 = tf.Variable(tf.zeros(1024))

        self.h2 = tf.Variable(tf.truncated_normal((1024, len(self.tp.vocab))))
        b2 = tf.Variable(tf.zeros(len(self.tp.vocab)))

        X = tf.nn.embedding_lookup(embeddings, pX)
        X = tf.reshape(X, [-1,self.embedding_size*self.tp.sequence_length])
        a1 = tf.matmul(X, self.h1) + b1
        y1 = tf.nn.relu(a1)
        logits = tf.matmul(y1,self.h2) + b2

        #return [logits, regularizers]
        return logits


    def loss(self, logits, pY):
        loss_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, pY)
        mean_loss = tf.reduce_mean(loss_value)# + \
                    #(0.005*tf.nn.l2_loss(self.h1)) + \
                    #(0.005*tf.nn.l2_loss(self.h2))
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
        print logits
        print char
        return char
