import numpy as np
import nltk
from threading import Thread
import Queue
import time


class ParsingWorker(Thread):
    def __init__(self, textparser):
        Thread.__init__(self)
        self.queue = textparser.q
        self.vocab = textparser.vocab
        self.tp = textparser

    def run(self):
        while True:
            split_name, idx, batch = self.queue.get()
            X = np.zeros((len(batch)-1,),dtype=np.int32)
            Y = np.zeros((self.tp.batch_size,),dtype=np.int32)
            for i, char in enumerate(batch[:-1]):
                if char not in self.vocab:
                    X[i] = 0
                    continue
                X[i] = self.vocab[char]
            X = X.reshape(self.tp.batch_size, self.tp.sequence_length)
            Y[:-1] = X[1:, 0]
            Y[-1] = self.vocab[batch[-1]]

            self.tp.batches[split_name][idx] = (X,Y)






class TextParser(Thread):
    def __init__(self, path, splits=[0.8,0.1,0.1]):
        Thread.__init__(self)
        self.vocab = self.get_character_set(path)
        self.path = path
        assert np.sum(splits) == 1
        self.splits = np.cumsum(splits)
        self.get_split_idx()
        self.active_split = 'train'
        self.do_seek = True
        self.batches = {'train' : {}, 'cv' : {}, 'test' : {}}
        self.max_idx = {'train' : 99999999999, 'cv' : 9999999999, 'test' : 9999999999}
        self.epoch = 1


    def get_split_idx(self):
        with open(self.path) as f:
            for n, line in enumerate(f):
                pass
        self.n = n

        bounds = np.array([0] + (np.round(self.splits*n)).tolist(), dtype=np.int32)
        self.seek_idx = []
        with open(self.path) as f:
            for i, line in enumerate(f):
                if i == bounds[0]: self.seek_idx.append(f.tell())
                if i == bounds[1]: self.seek_idx.append(f.tell())
                if i == bounds[2]: self.seek_idx.append(f.tell())

        self.bounds = {
                       'train' :  (bounds[0], bounds[1], self.seek_idx[0]),
                       'cv' :  (bounds[1], bounds[2], self.seek_idx[1]),
                       'test' :  (bounds[2], bounds[3], self.seek_idx[2])
                      }

    def get_character_set(self, path, char_limit=100000):
        vocab = {}
        idx = 1
        char_count = 0
        with open(path) as f:
            for line in f:
                for char in line.lower():
                    char_count+=1
                    if char_count > char_limit: return vocab
                    if char not in vocab:
                        vocab[char] = idx
                        idx+=1
        return vocab

    def prepare_batches(self, batch_size, sequence_length, workers=4, max_queue_length = 20):
        self.q = Queue.Queue()
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.max_queue_length = max_queue_length
        for i in range(workers):
            t = ParsingWorker(self)
            t.daemon = True
            t.start()

        self.daemon = True
        self.start()

    def switch_split(self, split_name='train'):
        self.do_seek = True
        self.active_split = split_name
        self.samples = self.n/self.batch_size*self.sequence_length
        self.current_idx = 0

    def get_next_feed_dict(self, pX, pY):
        if self.current_idx not in self.batches[self.active_split]:
            while self.current_idx not in self.batches[self.active_split]:
                time.sleep(0.01)
                if self.current_idx > self.max_idx[self.active_split]: self.current_idx = 0
        batchX, batchY = self.batches[self.active_split].pop(self.current_idx, None)
        self.current_idx+=1

        return { pX : batchX, pY : batchY }


    def run(self):
        needed_size = self.batch_size*self.sequence_length
        while True:
            with open(self.path) as f:
                f.seek(self.bounds[self.active_split][2])
                lines = []
                length = 0
                self.do_seek = False
                idx = 0
                self.current_idx = 0
                for line in f:
                    lines.append(line)
                    length += len(line)
                    if length >= needed_size+1:
                        while (len(self.batches) > self.max_queue_length
                                or self.q.qsize() > self.max_queue_length):
                            time.sleep(0.1)
                        s = "".join(lines).lower()
                        if self.do_seek: break

                        self.q.put((self.active_split, idx, s[:needed_size+1]))
                        idx+=1
                        del lines[:]
                        lines.append(s[needed_size:])
                        length = len(lines[0])
                if not self.do_seek:
                    self.max_idx[self.active_split] = idx-1
                    if self.active_split == 'train':
                        print 'EPOCH: {0}'.format(self.epoch)
                        self.epoch+=1




