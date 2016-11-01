import numpy as np
import nltk
from threading import Thread
import Queue
import time


class ParsingWorker(Thread):
    def __init__(self, queue, results, vocab):
        Thread.__init__(self)
        self.queue = queue
        self.results = results
        self.vocab = vocab

    def run(self):
        while True:
            batch = self.queue.get()
            X = np.zeros((len(batch),),dtype=np.int32)
            #print batch
            for i, char in enumerate(batch):
                if char not in self.vocab:
                    X[i] = 0
                    continue
                X[i] = self.vocab[char]


            self.results.put(X)






class TextParser(Thread):
    def __init__(self, path):
        Thread.__init__(self)
        self.vocab = self.get_character_set(path)
        self.path = path


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
        self.results = Queue.Queue()
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.max_queue_length = max_queue_length
        for i in range(workers):
            t = ParsingWorker(self.q, self.results, self.vocab)
            t.daemon = True
            t.start()

        self.daemon = True
        self.start()


    def run(self):
        print 'running thread'
        needed_size = self.batch_size*self.sequence_length
        lines = []
        length = 0
        while True:
            with open(self.path) as f:
                for line in f:
                    lines.append(line)
                    length += len(line)
                    if length >= needed_size:
                        while self.results.qsize() > self.max_queue_length:
                            time.sleep(0.1)
                        s = "".join(lines).lower()
                        self.q.put(s[:needed_size])
                        del lines[:]
                        lines.append(s[needed_size:])
                        length = len(lines[0])



