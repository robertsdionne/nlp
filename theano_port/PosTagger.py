
class PosTagger(object):

  def train(self, tagged_sentences, learning_rate, iterations):
    raise NotImplementedError()

  def validate(self, tagged_sentences):
    raise NotImplementedError()

  def tag(self, sentence):
    raise NotImplementedError()

  def score_tagging(self, tagged_sentence):
    raise NotImplementedError()
