from theano_port.PosTagger import PosTagger

class DumbPosTagger(PosTagger):
  
  def train(self, tagged_sentences, learning_rate, iterations):
    pass

  def validate(self, tagged_sentences):
    pass

  def tag(self, sentence):
    return ['NN' for i in xrange(0, len(sentence))]

  def score_tagging(self, tagged_sentence):
    return float('-inf')
