import unittest
from theano_port.DataLoader import DataLoader

class TestDataLoader(unittest.TestCase):

  def setUp(self):
    self.data_loader = DataLoader()

  def test_read_tagged_sentences(self):
    tagged_sentences, word_set, tags = self.data_loader.read_tagged_sentences(
        DataLoader.DEV_IN_DOMAIN_FILENAME)
    self.assertEqual('influential_JJ members_NNS of_IN the_DT house_NNP ways_NNP and_CC ' +
        'means_NNP committee_NNP introduced_VBD legislation_NN that_WDT would_MD restrict_VB ' +
        'how_WRB the_DT new_JJ savings_NNS -_HYPH and_CC -_HYPH loan_NN bailout_NN agency_NN ' +
        'can_MD raise_VB capital_NN ,_, creating_VBG another_DT potential_JJ obstacle_NN to_IN ' +
        'the_DT government_NN \'s_POS sale_NN of_IN sick_JJ thrifts_NNS ._.',
        str(tagged_sentences[0]))

  def test_tokenize_numbers(self):
    self.assertEqual('0/0/0', self.data_loader.tokenize_numbers(
        '+123.456,999.123/-123.123,999,123/+333.9999,888'))
    self.assertEqual('0', self.data_loader.tokenize_numbers('50,000'))
    self.assertEqual('0th', self.data_loader.tokenize_numbers('-50000.000,0000th'))

if __name__ == '__main__':
  unittest.main()
