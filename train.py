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
        self.pGenXBeam = tf.placeholder(tf.int32, shape=(2**4, self.tp.sequence_length))

    def init_model(self):
       self.model = model.Model(self.embedding_size, self.tp)
       self.logits= self.model.prediction(self.pX)
       self.mean_loss = self.model.loss(self.logits, self.pY)
       self.opt_step = self.model.optimize(self.mean_loss, self.lr)
       self.evaluate = self.model.evaluation(self.logits, self.pY)
       self.logitsGen = self.model.prediction(self.pGenX)
       self.generate = self.model.generate(self.logitsGen, 1)

       self.logitsGenBeam = self.model.prediction(self.pGenXBeam)
       self.generate_beam = self.model.generate(self.logitsGenBeam, 2)


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

    def generate_text(self, seed_text, kmax=2, levels=4, sequence_length=150):
        sequence = self.tp.get_sequence_from_text(seed_text)
        X = np.zeros((1,self.tp.sequence_length))
        if sequence.shape[0] > self.tp.sequence_length:
            X[0, :] = sequence[-self.tp.sequence_length]
        else:
            X[0, -sequence.shape[0]:] = sequence

        for i in range(sequence_length):
            feed_dict= { self.pGenX : X}
            next_char, softmax = self.sess.run(self.generate, feed_dict)
            next_char = next_char[0][0]
            X[0,:-1] = X[0,1:]
            X[0, -1] = next_char
            if next_char != 0:
                seed_text += self.tp.idx2vocab[next_char]

        print seed_text

    def generate_text_with_beam(self, seed_text, kmax=2, levels=4, sequence_length=150):
        sequence = self.tp.get_sequence_from_text(seed_text)
        X = np.zeros((1,self.tp.sequence_length))
        if sequence.shape[0] > self.tp.sequence_length:
            X[0, :] = sequence[-self.tp.sequence_length]
        else:
            X[0, -sequence.shape[0]:] = sequence

        m = self.tp.sequence_length
        n = kmax**levels

        predictions = []
        inputs = np.zeros((n,m))
        softmax_scores = np.zeros((n,levels))
        for i in range(sequence_length):
            inputs[:] = X[0,:]
            next_ids = []
            for i in range(levels):
                spacing = n/(kmax**(i))
                next_idx = [k for k in range(n) if k % spacing == 0][1:]
                next_idx.append(n)
                prev_idx = 0
                for idx in next_idx:
                    #kargmax = np.random.randint(0,60, kmax).tolist()
                    feed_dict= { self.pGenXBeam : inputs}
                    kargmax, softmax = self.sess.run(self.generate_beam, feed_dict)
                    kargmax = kargmax[prev_idx]
                    softmax = softmax[prev_idx]
                    vec = np.repeat(kargmax,spacing/kmax)
                    inputs[prev_idx:idx,:-1] = inputs[prev_idx:idx, 1:]
                    inputs[prev_idx:idx,-1] = vec

                    vec = np.repeat(softmax,spacing/kmax)
                    softmax_scores[prev_idx:idx,i] = vec
                    prev_idx = idx

            bestidx = np.argmax(np.prod(softmax_scores,1))
            best_char = inputs[bestidx][-levels]
            X[0,:-1] = X[0,1:]
            X[0,-1] = best_char
            seed_text+= self.tp.idx2vocab[best_char]
        print seed_text






