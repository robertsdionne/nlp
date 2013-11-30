require 'torch'
require 'nn'

dofile 'Asserts.lua'
dofile 'TaggedSentence.lua'

-- Test nn.TaggedSentence.
local words = {'one', 'two', 'three'}
local tags = {'TAG1', 'TAG2', 'TAG3'}
local sentence = nn.TaggedSentence(words, tags)
assertEquals('one_TAG1 two_TAG2 three_TAG3', tostring(sentence))

-- Test nn.TaggedSentence throws an error for mismatched word and tag table sizes.
words = {'one', 'two'}
tags = {'TAG1'}
local success, message = pcall(nn.TaggedSentence, words, tags)
assert(not success)
assert('TaggedSentence.lua:5: Word count must match tag count!' == message)
