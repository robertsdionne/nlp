require 'torch'
require 'nn'

dofile 'LoadedLookupTable.lua'
dofile 'TaggedSentence.lua'
dofile 'DataLoader.lua'

-- Test loadData() to ensure it reads the entire data file.
local tagged_sentences, vocabulary, tags = loadData(TRAIN_FILENAME)

-- The training data has 39815 examples.
assert(39815 == #tagged_sentences)

-- The first tagged sentence is:
assert("In_IN an_DT Oct._NNP 19_CD review_NN of_IN ``_`` The_DT Misanthrope_NN ''_'' at_IN " ..
    "Chicago_NNP 's_POS Goodman_NNP Theatre_NNP (_( ``_`` Revitalized_VBN Classics_NNS Take_VBP " ..
    "the_DT Stage_NN in_IN Windy_NNP City_NNP ,_, ''_'' Leisure_NN &_CC Arts_NNS )_) ,_, the_DT " ..
    "role_NN of_IN Celimene_NNP ,_, played_VBN by_IN Kim_NNP Cattrall_NNP ,_, was_VBD " ..
    "mistakenly_RB attributed_VBN to_IN Christina_NNP Haag_NNP ._." ==
        tagged_sentences[1]:toString())

-- The last tagged sentence is:
assert("That_DT could_MD cost_VB him_PRP the_DT chance_NN to_TO influence_VB the_DT outcome_NN " ..
  "and_CC perhaps_RB join_VB the_DT winning_VBG bidder_NN ._." ==
      tagged_sentences[39815]:toString())

assert(40366 == #vocabulary)
assert(48 == #tags)

-- local function tokenizeNumber(number)
--   local state = 0
--   local size = 0
--   local idx = 1
--   local finished = false

--   while not finished do
--     local c = number:sub(idx, idx)
--     idx = idx + 1

--     if '' == c then
--       break
--     end

--     if 0 == state then
--       if c == '+' or c == '-' then
--         state = 1
--       elseif c == '.' or c == ',' then
--         state = 2
--       elseif c:match('%d') then
--         state = 4
--       else
--         finished = true
--       end
--     elseif 1 == state then
--       if c == '.' or c == ',' then
--         state = 2
--       elseif c:match('%d') then
--         state = 4
--       else
--         finished = true
--       end
--     elseif 2 == state then
--       if c:match('%d') then
--         state = 4
--       else
--         finished = true
--       end
--     elseif 3 == state then
--       if c:match('%d') then
--         state = 4
--       else
--         finished = true
--       end
--     elseif 4 == state then
--       size = idx - 1
--       if c == '.' or c == ',' then
--         state = 3
--       elseif c:match('%d') then
--         state = 4
--       else
--         finished = true
--       end
--     end
--   end

--   return size
-- end

local table = nn.LoadedLookupTable.load()

for _, word in pairs(vocabulary) do
 word = word:lower()
 word = word:gsub('[0-9]+', '0')
 local index = table:queryIndex(word)
 if nil == index then
   index = 'UNKNOWN'
 end
 print(word .. ': ' .. index)
end
