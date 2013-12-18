import re
from TaggedSentence import TaggedSentence

class DataLoader(object):

  DEV_IN_DOMAIN_FILENAME = '../data/en-wsj-dev.pos'
  DEV_OUT_OF_DOMAIN_FILENAME = '../data/en-web-weblogs-dev.pos'
  TEST_FILENAME = '../data/en-web-test.tagged'
  TRAIN_FILENAME = '../data/en-wsj-train.pos'

  def read_tagged_sentences(self, filename, sentence_count = -1):
    data_file = open(filename)
    sentence_words = []
    sentence_tags = []
    tagged_sentences = []
    word_set = set()
    tag_set = set()
    maximum_sentence_length = 0
    total_sentence_length = 0
    for line in data_file.readlines():
      if not re.match('\\s', line):
        word, tag = line.split()
        word = self.tokenize_numbers(word.lower())
        sentence_words.append(word)
        sentence_tags.append(tag)
        word_set.add(word)
        tag_set.add(tag)
      else:
        tagged_sentence = TaggedSentence(sentence_words, sentence_tags)
        total_sentence_length += len(tagged_sentence)
        if maximum_sentence_length < len(tagged_sentence):
          maximum_sentence_length = len(tagged_sentence)
        tagged_sentences.append(tagged_sentence)
        sentence_words = []
        sentence_tags = []
    data_file.close()
    return tagged_sentences, word_set, list(tag_set), maximum_sentence_length, \
        total_sentence_length / len(tagged_sentences)

  def tokenize_numbers(self, string):
    return re.sub('[\\+\\-]?[0-9]+([\\.,][0-9]*)*', '0', string)
