
class Evaluator(object):

  def evaluate_tagger(self, pos_tagger, tagged_sentences, training_vocabulary, verbose = False):
    num_tags = 0.0
    num_tags_correct = 0.0
    num_unknown_words = 1.0
    num_unknown_words_correct = 1.0
    num_decoding_inversions = 0.0
    i = 0
    for tagged_sentence in tagged_sentences:
      if i % 100 == 0:
        print('finished ' + i + ' sentences / ' + len(tagged_sentences))
      i += 1
      words = tagged_sentence.words
      gold_tags = tagged_sentence.tags
      guessed_tags = pos_tagger.tag(tagged_sentence)
      for position in xrange(0, len(words)):
        word = words[position]
        gold_tag = gold_tags[position]
        guessed_tag = guessed_tags[position]
        if guessed_tag == gold_tag:
          num_tags_correct += 1.0
        num_tags += 1.0
        if word not in training_vocabulary:
          if guessed_tag == gold_tag:
            num_unknown_words_correct += 1.0
          num_unknown_words += 1.0
    print('  Tag Accuracy: ' + (num_tags_correct / num_tags))
    print('  (Unknown Accuracy: ' + (num_unknown_words_correct / num_unknown_words) + ')')
