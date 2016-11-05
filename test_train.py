from text_parser import TextParser
import train

path = '/home/tim/Dropbox/Notes/journal.txt'

tp = TextParser(path)
tp.prepare_batches(128,32)
trainer = train.Trainer(tp, 256)
trainer.train(50000)
trainer.generate_text('I think that')

