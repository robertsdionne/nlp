dofile 'Asserts.lua'
dofile 'DataLoader.lua'

local data_loader = nn.DataLoader()

-- Test loadData() to ensure it reads the entire data file.
local tagged_sentences = data_loader:readTaggedSentences(DEV_IN_DOMAIN_FILENAME)

-- The dev in-domain data has 1700 examples.
assertEquals(1700, #tagged_sentences)

assertEquals('influential_JJ members_NNS of_IN the_DT house_NNP ways_NNP and_CC means_NNP ' ..
    'committee_NNP introduced_VBD legislation_NN that_WDT would_MD restrict_VB how_WRB the_DT ' ..
    'new_JJ savings_NNS -_HYPH and_CC -_HYPH loan_NN bailout_NN agency_NN can_MD raise_VB ' ..
    'capital_NN ,_, creating_VBG another_DT potential_JJ obstacle_NN to_IN the_DT government_NN ' ..
    '\'s_POS sale_NN of_IN sick_JJ thrifts_NNS ._.', tostring(tagged_sentences[1]))

-- Test the number tokenizer.
assertEquals('0/0/0', data_loader:tokenizeNumbers(
      '+123.456,999.123/-123.123,999,123/+333.9999,888'))
assertEquals('0', data_loader:tokenizeNumbers('50,000'))
assertEquals('0th', data_loader:tokenizeNumbers('-50000.000,0000th'))
