require 'torch'
require 'nn'

local PosTagger = torch.class('nn.PosTagger')

-- Abstract base class for our PosTagger
function PosTagger:__init()
end

function PosTagger:train(taggedSentences)
  error('Not yet implemented!')
end

function PosTagger:validate(taggedSentences)
  error('Not yet implemented!')
end

function PosTagger:tag(sentence)
  error('Not yet implemented!')
end

function PosTagger:scoreTagging(taggedSentence)
  error('Not yet implemented!')
end
