dofile 'Asserts.lua'
dofile 'LoadedLookupTable.lua'

local lookupTable = nn.LoadedLookupTable.load()

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

--TODO(robertsdionne): fix backward and backwardUpdate
--print(lookupTable:backwardUpdate('reply', torch.rand(50, 50), 0.1))