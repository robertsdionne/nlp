import numpy

class LoadedLookupTable(object):

  # The path to the Collobert and Weston 2011 word embeddings file.
  EMBEDDINGS_FILENAME = '../embeddings/embeddings.txt'
  # The path to the Collobert and Weston 2011 words file.
  WORDS_FILENAME = '../embeddings/words.lst'

  PADDING = 1738
  UNKNOWN = 1739
  
  @classmethod
  def load(cls):
    embeddings = numpy.loadtxt(cls.EMBEDDINGS_FILENAME)
    words_file = open(cls.WORDS_FILENAME)
    word_to_index = {}
    index_to_word = {}
    index = 0
    for line in words_file.readlines():
      word = line.strip()
      word_to_index[word] = index
      index_to_word[index] = word
      index += 1
    words_file.close()
    return LoadedLookupTable(embeddings, word_to_index, index_to_word)

  def __init__(self, embeddings, word_to_index, index_to_word):
    self.embeddings = embeddings
    self.word_to_index = word_to_index
    self.index_to_word = index_to_word

  def forward(self, item):
    return self.embeddings[self.item_to_index(item), 0:]

  def item_to_index(self, item):
    if str == type(item):
      return self.query_index(item)
    elif int == type(item):
      return item

  def query_index(self, word):
    return self.word_to_index.get(word, self.word_to_index['UNKNOWN'])

  def query_word(self, index):
    return self.index_to_word.get(index, 'UNKNOWN')
