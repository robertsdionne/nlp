
class Evaluator(object):

  def evaluate_tagger(self, pos_tagger, tagged_sentences, training_vocabulary, verbose = False):
    num_tags = 0.0
    num_tags_correct = 0.0
    num_unknown_words = 1.0
    num_unknown_words_correct = 1.0
    num_decoding_inversions = 0.0
    for tagged_sentence in tagged_sentences:
      words = tagged_sentence.words
      gold_tags = tagged_sentence.tags
      guessed_tags = pos_tagger.tag(tagged_sentence)
      
