require 'torch'
require 'math'
require 'nn'
dofile 'CrossRNN.lua'
--dofile 'CrossRNNCVG.lua'

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

function RnnPosTagger:train(tagged_sentences, learningRate, iterations, evaluator, dev_out_tagged_sentences, training_vocabulary)
  learningRate = learningRate or 1
  iterations = iterations or 100
  -- iterations over corps
  for itr = 1,iterations do
      print("Began the iteration: ".. itr)
    -- iterations over sentence
    print("The number of sentences:"..#tagged_sentences);
    print();
    for i = 1, #tagged_sentences do
        if i % 100 == 0 then
            print("Finished "..i.." sentences.");
        end
        local currentSent = tagged_sentences[i]
        -- Genrate the tuples needed for 
        local represents, indexes, tagsId = {}, {}, {}
        for wn = 1, #currentSent.words do
            local word = currentSent['words'][wn]
            local represent = self.lookupTable:forward(word)[1]:clone()

            local index = self.lookupTable:queryIndex(word)
            local tagId = self.tagIndex[currentSent['tags'][wn]]
            table.insert(represents,represent)
            --print(represents[1])
            table.insert(indexes, index)
            table.insert(tagsId, tagId)
        end
        currentSent.represents = represents
        currentSent.index = indexes
        currentSent.tagsId = tagsId
        --print(currentSent) -- @WHY output something strange
        self.initRepresent = torch.rand(self.leftInputSize)--= self.lookupTable:forward(nn.LoadedLookupTable.PADDING)[1]
        -- forward the rnn
        self.rnn:forward(currentSent, self.initRepresent)
        -- backward the rnn
        self.rnn:backward(currentSent, self.initRepresent)
        -- update the parameters
        self.rnn:updateParameters(learningRate ) --/ math.sqrt(1 + iterations)
        --error('Implementing!')
    end
    collectgarbage()
    print('Validating on out-of-domain data:')
    evaluator:evaluateTagger(self, dev_out_tagged_sentences, training_vocabulary, false)
  end
end

function RnnPosTagger:validate(tagged_sentences)
  error('Not yet implemented!')
end

function RnnPosTagger:tag(sentence)
    local currentSent = sentence
    local represents, indexes, tagsId = {}, {}, {}
        for wn = 1, #currentSent.words do
            local word = currentSent['words'][wn]
            local represent = self.lookupTable:forward(word)[1]:clone()
            local index = self.lookupTable:queryIndex(word)
            local tagId = self.tagIndex[currentSent['tags'][wn]]
            table.insert(represents,represent)
            table.insert(indexes, index)
            table.insert(tagsId, tagId)
        end
        currentSent.represents = represents
        currentSent.index = indexes
        currentSent.tagsId = tagsId
        --print(currentSent) -- @WHY output something strange
        self.initRepresent = self.lookupTable:forward(nn.LoadedLookupTable.PADDING)[1]
        -- forward the rnn
        local tagsPredId = self.rnn:forward(currentSent, self.initRepresent)
        local tagsPredName = {}
        for t = 1, #tagsPredId do
            --print(tagsPredId[t])
            tagsPredName[t] = self.tagSet[tagsPredId[t]]
        end
        return tagsPredName;
end

function RnnPosTagger:scoreTagging(tagged_sentence)
    return 0;
  --error('Not yet implemented!')
end
