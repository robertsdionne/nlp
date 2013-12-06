dofile 'Asserts.lua'
dofile 'DumbPosTagger.lua'

local dumb_pos_tagger = nn.DumbPosTagger()

local success = pcall(dumb_pos_tagger.train, dumb_pos_tagger)
assertTrue(success)

local success = pcall(dumb_pos_tagger.validate, dumb_pos_tagger)
assertTrue(success)

local tags = dumb_pos_tagger:tag({'a', 'b', 'c'})
assertEquals(3, #tags)
for i = 1, #tags do
  assertEquals('NN', tags[i])
end

assertEquals(-math.huge, dumb_pos_tagger:scoreTagging({}))
