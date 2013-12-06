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
    self.rnn = nn.CrossRNN(self.leftInputSize, self.rightInputSize, #self.tagSet, self.lookupTable);
    -- Index the tags
    self:indexTag()
end

-- assign index to tag
function RnnPosTagger:indexTag()
    self.tagIndex = {}
    local index = 1
    for _,tag in pairs(self.tagSet) do
        self.tagIndex[tag] = index
        index = index + 1
    end
end

function RnnPosTagger:train(tagged_sentences, learningRate, iterations)
  learningRate = learningRate or 0.01
  iterations = iterations or 2
  -- iterations over corps
  for itr = 1,iterations do
      print("Begain the iteration: ".. itr)
    -- iterations over sentence
    for i = 1,#tagged_sentences do
        local currentSent = tagged_sentences[i]
        -- Genrate the tuples needed for 
        local represents, indexes, tagsId = {}, {}, {}
        for wn = 1,#currentSent.words do
            local word = currentSent['words'][wn]
            local represent = self.lookupTable:forward(word)[1]
            local index = self.lookupTable:queryIndex(word)
            local tagId = self.tagIndex[currentSent['tags'][wn]]
            table.insert(represents,represent)
            table.insert(indexes, index)
            table.insert(tagsId, tagId)
        end
        currentSent.represents = represents
        currentSent.index = indexes
        currentSent.tagsId = tagsId
        print(currentSent) -- @WHY output something strange
        local initRepresent = self.lookupTable:forward(nn.LoadedLookupTable.PADDING)[1]
        -- forward the rnn
        self.rnn:forward(currentSent, initRepresent)
        -- backward the rnn
        self.rnn:backward(currentSent, initRepresent)
        -- update the parameters
        self.rnn:updateParameters(learningRate)
        error('Implementing!')
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
