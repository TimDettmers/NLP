from text_parser import TextParser
import train

path = '/home/tim/Dropbox/Notes/journal.txt'

tp = TextParser(path)
tp.prepare_batches(512,24)
trainer = train.Trainer(tp, 256)
trainer.cv_interval = 200
trainer.train(2000)
trainer.generate_text('I think that')
#trainer.generate_text_with_beam('I think that')
trainer.generate_text('The whole weekend was a huge success. We won the hackathon which was great, but much better was my experience with Natalia the evening before the hackathon. Before this weekend, I did not know if I had conquered my social anxiety at the last conference, where I only realized at the end of the day, that I was not anxi')


