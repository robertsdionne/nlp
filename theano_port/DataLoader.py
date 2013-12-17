import re
from theano_port.TaggedSentence import TaggedSentence

class DataLoader(object):

  DEV_IN_DOMAIN_FILENAME = 'data/en-wsj-dev.pos'

  def read_tagged_sentences(self, filename, sentence_count = -1):
    data_file = open(filename)
    sentence_words = []
    sentence_tags = []
    tagged_sentences = []
    word_set = set()
    tag_set = set()
    for line in data_file.readlines():
      if not re.match('\\s', line):
        word, tag = line.split()
        word = self.tokenize_numbers(word.lower())
        sentence_words.append(word)
        sentence_tags.append(tag)
        word_set.add(word)
        tag_set.add(tag)
      else:
        tagged_sentences.append(TaggedSentence(sentence_words, sentence_tags))
        sentence_words = []
        sentence_tags = []
    data_file.close()
    return tagged_sentences, word_set, list(tag_set)

  def tokenize_numbers(self, string):
    return re.sub('[\\+\\-]?[0-9]+([\\.,][0-9]*)*', '0', string)
