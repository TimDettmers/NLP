from text_parser import TextParser
import time

path = '/home/tim/Dropbox/Notes/journal.txt'

tp = TextParser(path)
tp.prepare_batches(128,32)

print len(tp.vocab.keys())


for i in range(10):
    time.sleep(1)
    print tp.results.qsize()
    print tp.results.get().shape
