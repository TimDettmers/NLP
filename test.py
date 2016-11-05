from text_parser import TextParser
import time

path = '/home/tim/Dropbox/Notes/journal.txt'

tp = TextParser(path)
tp.prepare_batches(128,32, 4)

print len(tp.vocab.keys())


for i in range(10):
    tp.switch_split('cv')
    print tp.get_next_feed_dict('a', 'b')
    tp.switch_split('train')
    print tp.get_next_feed_dict('a', 'b')
    tp.switch_split('cv')
    print tp.get_next_feed_dict('a', 'b')
    tp.switch_split('train')
    print tp.get_next_feed_dict('a', 'b')


for i in range(1000000):
    tp.get_next_feed_dict('a', 'b')

