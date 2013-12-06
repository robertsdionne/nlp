require 'torch'
require 'nn'
dofile 'CrossRNN.lua'

local RnnPosTagger, parent = torch.class('nn.RnnPosTagger', 'nn.PosTagger')

-- Abstract base class for our PosTagger
function RnnPosTagger:__init(lookupTable, leftInputSize, rightInputSize, tagSet)
    self.lookupTable = lookupTable
    self.rightInputSize = rightInputSize
    self.tagSet = tagSet
    self.leftInputSize = leftInputSize 
    print("The embeding size is: ".. self.leftInputSize)
    print("The right size is: ".. self.rightInputSize)
    print("The number of tags are: ".. #self.tagSet)
    -- new CrossRNN
    self.rnn = nn.CrossRNN(self.leftInputSize, self.rightInputSize, #self.tagSet, self.lookUpTable);
end

function RnnPosTagger:train(tagged_sentences, learningRate, iterations)
  learningRate = learningRate or 0.01
  iterations = iterations or 2
  -- iterations over corps
  for itr = 1,iterations do
    -- iterations over sentence
    for i = 1,#tagged_sentences do
        local currentSent = tagged_sentences[i]
        -- Genrate the tuples needed for 
        local represents, indexes = {}, {}
        for wn = 1,#currentSent.words do
            local word = currentSent['words'][wn]
            local represent = self.lookupTable:forward(word)
            local index = self.lookupTable:queryIndex(word)
            table.insert(represents,represent)
            table.insert(indexes, index)
        end
        currentSent.represents = represents
        currentSent.indexes = indexes
        print(currentSent)
        error('Implementing!')
        -- forward the rnn
        -- backward the rnn
        -- update the parameters
    end
  end
  error('Implementing!')
end

function RnnPosTagger:validate(tagged_sentences)
  error('Not yet implemented!')
end

function RnnPosTagger:tag(sentence)
  error('Not yet implemented!')
end

function RnnPosTagger:scoreTagging(tagged_sentence)
  error('Not yet implemented!')
end
