dofile 'Asserts.lua'
dofile 'LoadedLookupTable.lua'

local lookupTable = nn.LoadedLookupTable.load()

local index = lookupTable:queryIndex('to')
print(lookupTable:forward(index))
print(lookupTable:forward('to'))
assertEquals(1739, nn.LoadedLookupTable.PADDING)
assertEquals(1740, nn.LoadedLookupTable.UNKNOWN)
assertEquals(nn.LoadedLookupTable.PADDING, lookupTable:queryIndex('PADDING'))
assertEquals(nn.LoadedLookupTable.UNKNOWN, lookupTable:queryIndex('UNKNOWN'))

assertEquals(1246, lookupTable:queryIndex('0/0/0'))
assertEquals(1643, lookupTable:queryIndex('0th'))
assertEquals(96969, lookupTable:queryIndex('reply'))
assertEquals(nn.LoadedLookupTable.UNKNOWN, lookupTable:queryIndex('wree'))
assertEquals(129976, lookupTable:queryIndex('zygmunt'))

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

zeroEmbedding = zeroLookupTable:forward('word')[1]
assertFloatEquals(-0.1, zeroEmbedding[1])
assertFloatEquals(-0.1, zeroEmbedding[2])
assertFloatEquals(-0.1, zeroEmbedding[3])
assertFloatEquals(-0.1, zeroEmbedding[4])
assertFloatEquals(-0.1, zeroEmbedding[5])

zeroLookupTable:reset(torch.Tensor(5, 5):zero())

-- reset weights
zeroEmbedding = zeroLookupTable:forward('word')[1]
assertFloatEquals(0, zeroEmbedding[1])
assertFloatEquals(0, zeroEmbedding[2])
assertFloatEquals(0, zeroEmbedding[3])
assertFloatEquals(0, zeroEmbedding[4])
assertFloatEquals(0, zeroEmbedding[5])

-- back propagate
zeroLookupTable:backward('word', torch.Tensor(5, 5):fill(1))
zeroLookupTable:updateParameters(0.1)

zeroEmbedding = zeroLookupTable:forward('word')[1]
assertFloatEquals(-0.1, zeroEmbedding[1])
assertFloatEquals(-0.1, zeroEmbedding[2])
assertFloatEquals(-0.1, zeroEmbedding[3])
assertFloatEquals(-0.1, zeroEmbedding[4])
assertFloatEquals(-0.1, zeroEmbedding[5])
