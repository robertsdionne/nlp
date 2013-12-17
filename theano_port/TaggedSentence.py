
class TaggedSentence(object):
  
  def __init__(self, words, tags):
    if len(words) != len(tags):
      raise ValueError('Word count must match tag count!')
    self.words = words
    self.tags = tags

  def __len__(self):
    return len(self.words)

  def __str__(self):
    return ' '.join([self.words[i] + '_' + self.tags[i] for i in xrange(0, len(self.words))])
