require 'torch'
require 'nn'

local PosTagger = torch.class('nn.PosTagger')

-- Abstract base class for our PosTagger
function PosTagger:__init()
end

function PosTagger:train(tagged_sentences)
  error('Not yet implemented!')
end

function PosTagger:validate(tagged_sentences)
  error('Not yet implemented!')
end

function PosTagger:tag(sentence)
  error('Not yet implemented!')
end

function PosTagger:scoreTagging(tagged_sentence)
  error('Not yet implemented!')
end
