dofile 'Asserts.lua'
dofile 'LoadedLookupTable.lua'

local lookupTable = nn.LoadedLookupTable.load()

-- Test the index and word reach the same location in the table
local embedding_from_index = lookupTable:forward(lookupTable:queryIndex('to'))[1]
local embedding_from_word = lookupTable:forward('to')[1]

assertEquals(embedding_from_index:size(1), embedding_from_word:size(1))
for i = 1, embedding_from_index:size(1) do
  assertEquals(embedding_from_index[i], embedding_from_word[i])
end

-- Test known indices
assertEquals(1739, nn.LoadedLookupTable.PADDING)
assertEquals(1740, nn.LoadedLookupTable.UNKNOWN)
assertEquals(nn.LoadedLookupTable.PADDING, lookupTable:queryIndex('PADDING'))
assertEquals(nn.LoadedLookupTable.UNKNOWN, lookupTable:queryIndex('UNKNOWN'))

assertEquals(1246, lookupTable:queryIndex('0/0/0'))
assertEquals(1643, lookupTable:queryIndex('0th'))
assertEquals(96969, lookupTable:queryIndex('reply'))
assertEquals(nn.LoadedLookupTable.UNKNOWN, lookupTable:queryIndex('wree'))
assertEquals(129976, lookupTable:queryIndex('zygmunt'))

-- Test known embeddings
local embedding1 = lookupTable:forward(1)[1]
assertFloatEquals(-1.03682, embedding1[1])
assertFloatEquals(1.5799, embedding1[5])
assertFloatEquals(-0.64853, embedding1[50])

local embedding130000 = lookupTable:forward(130000)[1]
assertFloatEquals(-0.238824, embedding130000[1])
assertFloatEquals(-1.76695, embedding130000[5])
assertFloatEquals(-1.4733, embedding130000[50])

-- Test back propagation
local zeroLookupTable = nn.LoadedLookupTable(
    torch.Tensor(5, 5):zero(), {['word'] = 1}, {[1] = 'word'})

local zeroEmbedding = zeroLookupTable:forward('word')[1]
assertFloatEquals(0, zeroEmbedding[1])
assertFloatEquals(0, zeroEmbedding[2])
assertFloatEquals(0, zeroEmbedding[3])
assertFloatEquals(0, zeroEmbedding[4])
assertFloatEquals(0, zeroEmbedding[5])

-- back propagate
zeroLookupTable:backwardUpdate('word', torch.Tensor(5, 5):fill(1), 0.1)

local zeroEmbedding = zeroLookupTable:forward('word')[1]
assertFloatEquals(-0.1, zeroEmbedding[1])
assertFloatEquals(-0.1, zeroEmbedding[2])
assertFloatEquals(-0.1, zeroEmbedding[3])
assertFloatEquals(-0.1, zeroEmbedding[4])
assertFloatEquals(-0.1, zeroEmbedding[5])

zeroLookupTable:reset(torch.Tensor(5, 5):zero())

-- reset weights
local zeroEmbedding = zeroLookupTable:forward('word')[1]
assertFloatEquals(0, zeroEmbedding[1])
assertFloatEquals(0, zeroEmbedding[2])
assertFloatEquals(0, zeroEmbedding[3])
assertFloatEquals(0, zeroEmbedding[4])
assertFloatEquals(0, zeroEmbedding[5])

-- back propagate
zeroLookupTable:backward('word', torch.Tensor(5, 5):fill(1))
zeroLookupTable:updateParameters(0.1)

local zeroEmbedding = zeroLookupTable:forward('word')[1]
assertFloatEquals(-0.1, zeroEmbedding[1])
assertFloatEquals(-0.1, zeroEmbedding[2])
assertFloatEquals(-0.1, zeroEmbedding[3])
assertFloatEquals(-0.1, zeroEmbedding[4])
assertFloatEquals(-0.1, zeroEmbedding[5])
