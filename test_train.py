from text_parser import TextParser
import train

path = '/home/tim/Dropbox/Notes/journal.txt'

tp = TextParser(path)
tp.prepare_batches(512,24)
trainer = train.Trainer(tp, 128)
trainer.cv_interval = 1000
trainer.train(7000)
trainer.generate_text('I think that')
trainer.generate_text('The whole weekend was a huge success. We won the hackathon which was great, but much better was my experience with Natalia the evening before the hackathon. Before this weekend, I did not know if I had conquered my social anxiety at the last conference, where I only realized at the end of the day, that I was not anxi')


