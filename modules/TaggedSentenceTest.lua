require 'torch'
require 'nn'

dofile 'TaggedSentence.lua'

words = {'one', 'two', 'three'}
tags = {'TAG1', 'TAG2', 'TAG3'}
sentence = nn.TaggedSentence(words, tags)
assert('one_TAG1 two_TAG2 three_TAG3' == sentence:toString())
