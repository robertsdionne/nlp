from TaggedSentence import TaggedSentence
import unittest

class TaggedSentenceTest(unittest.TestCase):

  def test_tagged_sentence(self):
    words = ['one', 'two', 'three']
    tags = ['TAG1', 'TAG2', 'TAG3']
    sentence = TaggedSentence(words, tags)
    self.assertEquals('one_TAG1 two_TAG2 three_TAG3', str(sentence))

  def test_mismatched_tag_count(self):
    words = ['one', 'two']
    tags = ['TAG1']
    self.assertRaises(ValueError, TaggedSentence, words, tags)

if __name__ == '__main__':
  unittest.main()
