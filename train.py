import tensorflow as tf
import model
import numpy as np


class Trainer(object):
    def __init__(self, text_parser, embedding_size):
        self.tp = text_parser
        self.embedding_size = embedding_size
        self.lr = 0.001
        self.cv_interval = 1000
        self.L2 = 0.00005

    def init_placeholders(self):
        self.pX = tf.placeholder(tf.int32, shape=(self.tp.batch_size, self.tp.sequence_length))
        self.pY = tf.placeholder(tf.int32, shape=(self.tp.batch_size,))
        self.pGenX = tf.placeholder(tf.int32, shape=(1, self.tp.sequence_length))

    def init_model(self):
       self.model = model.Model(self.embedding_size, self.tp)
       self.logits= self.model.prediction(self.pX)
       self.mean_loss = self.model.loss(self.logits, self.pY)
       self.opt_step = self.model.optimize(self.mean_loss, self.lr)
       self.evaluate = self.model.evaluation(self.logits, self.pY)
       self.logitsGen = self.model.prediction(self.pGenX)
       self.generate = self.model.generate(self.logitsGen)


    def get_errors(self, split_name):
        self.tp.switch_split(split_name)
        correct = 0
        idx = 0
        while idx < self.tp.max_idx[split_name]:
            feed_dict = self.tp.get_next_feed_dict(self.pX, self.pY)
            correct += self.sess.run(self.evaluate, feed_dict)
            idx+=1
        return np.round(1.0 - (correct/(1.0*self.tp.max_idx[split_name]*self.tp.batch_size)),4)

    def train(self, batch_count):
            with tf.Graph().as_default():
                self.init_placeholders()
                self.init_model()
                init = tf.initialize_all_variables()
                self.sess = tf.Session()
                self.sess.run(init)
                for i in range(batch_count):
                    feed_dict = self.tp.get_next_feed_dict(self.pX,self.pY)
                    #print feed_dict[self.pY]
                    #print self.sess.run(self.generate_train, feed_dict)
                    #print self.logits
                    loss_value, _ = self.sess.run([self.mean_loss, self.opt_step], feed_dict)

                    if i % self.cv_interval == 0 and i > 0:
                        print 'Batch number: {0}'.format(i)
                        error_train = self.get_errors('train')
                        error_cv = self.get_errors('cv')
                        print "Train error: {0}".format(error_train)
                        print "Cross validation error: {0}".format(error_cv)
                        print "Loss value: {0}:".format(loss_value)
                        error_train = self.get_errors('train')

    def generate_text(self, seed_text, sequence_length=150):
        sequence = self.tp.get_sequence_from_text(seed_text)
        X = np.zeros((1,self.tp.sequence_length))
        if sequence.shape[0] > self.tp.sequence_length:
            X[0, :] = sequence[-self.tp.sequence_length]
        else:
            X[0, -sequence.shape[0]:] = sequence

        for i in range(sequence_length):
            feed_dict= { self.pGenX : X}
            next_char = self.sess.run(self.generate, feed_dict)[-1]
            X[0,:-1] = X[0,1:]
            X[0, -1] = next_char
            if next_char != 0:
                seed_text += self.tp.idx2vocab[next_char]

        print seed_text






