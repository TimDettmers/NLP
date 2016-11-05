import tensorflow as tf
import model
import numpy as np


class Trainer(object):
    def __init__(self, text_parser, embedding_size):
        self.tp = text_parser
        self.embedding_size = embedding_size
        self.lr = 0.001

    def init_placeholders(self):
        self.pX = tf.placeholder(tf.int32, shape=(self.tp.batch_size, self.tp.sequence_length))
        self.pY = tf.placeholder(tf.int32, shape=(self.tp.batch_size,))

    def init_model(self):
       self.model = model.Model(self.embedding_size, self.tp)
       self.logits = self.model.prediction(self.pX)
       self.mean_loss = self.model.loss(self.logits, self.pY)
       self.opt_step = self.model.optimize(self.mean_loss, self.lr)
       self.evaluate = self.model.evaluation(self.logits, self.pY)

    def get_errors(self, sess, split_name):
        self.tp.switch_split(split_name)
        correct = 0
        idx = 0
        while idx < self.tp.max_idx[split_name]:
            feed_dict = self.tp.get_next_feed_dict(self.pX, self.pY)
            correct += sess.run(self.evaluate, feed_dict)
            idx+=1
        return np.round(1.0 - (correct/(1.0*self.tp.max_idx[split_name]*self.tp.batch_size)),4)

    def train(self, batch_count):
            with tf.Graph().as_default():
                self.init_placeholders()
                self.init_model()
                init = tf.initialize_all_variables()
                sess = tf.Session()
                sess.run(init)
                for i in range(batch_count):
                    feed_dict = self.tp.get_next_feed_dict(self.pX,self.pY)
                    loss_value, _ = sess.run([self.mean_loss, self.opt_step], feed_dict)

                    if i % 1000 == 0 and i > 0:
                        error_train = self.get_errors(sess, 'train')
                        error_cv = self.get_errors(sess, 'cv')
                        print "Train error: {0}".format(error_train)
                        print "Cross validation error: {0}".format(error_cv)
                        print "Loss value: {0}:".format(loss_value)
                        error_train = self.get_errors(sess, 'train')


