require 'torch'
require 'math'
require 'nn'

dofile 'PosTagger.lua'

local DumbPosTagger, parent = torch.class('nn.DumbPosTagger', 'nn.PosTagger')

-- A dumb PosTagger that just returns {'NN', 'NN', ...} and -infinity.
function DumbPosTagger:__init()
end

function DumbPosTagger:train(tagged_sentences)
  -- do nothing
end

function DumbPosTagger:validate(tagged_sentences)
  -- do nothing
end

function DumbPosTagger:tag(sentence)
  local tags = {}
  for i = 1, #sentence do
    tags[i] = 'NN'
  end
  return tags
end

function DumbPosTagger:scoreTagging(tagged_sentence)
  return -math.huge
end
