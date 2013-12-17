from DumbPosTagger import DumbPosTagger
from TaggedSentence import TaggedSentence
import unittest

class DumbPosTaggerTest(unittest.TestCase):

  def setUp(self):
    self.dumb_pos_tagger = DumbPosTagger()

  def test_train(self):
    try:
      self.dumb_pos_tagger.train([], 0.1, 1)
    except NotImplementedError:
      self.fail('train() failed')

  def test_validate(self):
    try:
      self.dumb_pos_tagger.validate([])
    except NotImplementedError:
      self.fail('validate() failed')

  def test_tag(self):
    tags = self.dumb_pos_tagger.tag(['a', 'b', 'c'])
    self.assertEqual(3, len(tags))
    self.assertEqual(['NN', 'NN', 'NN'], tags)

  def test_score_tagging(self):
    self.assertEqual(float('-inf'), self.dumb_pos_tagger.score_tagging(TaggedSentence([], [])))

if __name__ == '__main__':
  unittest.main()
