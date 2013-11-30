require 'torch'
require 'nn'

local Evaluator = torch.class('nn.Evaluator')

function Evaluator:__init()
end

function Evaluator:evaluateTagger(pos_tagger, tagged_sentences, vocabulary)
end
