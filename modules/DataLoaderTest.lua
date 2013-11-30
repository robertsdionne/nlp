dofile 'Asserts.lua'
dofile 'DataLoader.lua'

local data_loader = nn.DataLoader()

-- Test loadData() to ensure it reads the entire data file.
local tagged_sentences = data_loader:readTaggedSentences(DEV_IN_DOMAIN_FILENAME)

-- The dev in-domain data has 1700 examples.
assertEquals(1700, #tagged_sentences)

-- Test the number tokenizer.
assertEquals('0/0/0', data_loader:tokenizeNumbers(
      '+123.456,999.123/-123.123,999,123/+333.9999,888'))
assertEquals('0', data_loader:tokenizeNumbers('50,000'))
assertEquals('0th', data_loader:tokenizeNumbers('-50000.000,0000th'))
