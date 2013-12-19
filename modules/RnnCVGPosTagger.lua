require 'torch'
require 'math'
require 'nn'
--dofile 'CrossRNN.lua'
dofile 'CrossRNNCVG.lua'

local RnnCVGPosTagger, parent = torch.class('nn.RnnCVGPosTagger', 'nn.PosTagger')

-- Abstract base class for our PosTagger
function RnnCVGPosTagger:__init(lookupTable, leftInputSize, rightInputSize, tagSet )
    self.lookupTable = lookupTable
    self.rightInputSize = rightInputSize
    self.tagSet = tagSet
    self.leftInputSize = leftInputSize 
    print("The embeding size is: ".. self.leftInputSize)
    print("The right size is: ".. self.rightInputSize)
    print("The number of tags are: ".. #self.tagSet)
    -- new CrossRNN
    self.rnn = nn.CrossRNNCVG(self.leftInputSize, self.rightInputSize, #self.tagSet, self.lookupTable);
    -- Index the tags
    self:indexTag()
end

-- assign index to tag
function RnnCVGPosTagger:indexTag()
    self.tagIndex = {}
    local index = 1
    for _,tag in pairs(self.tagSet) do
        self.tagIndex[tag] = index
        index = index + 1
    end
end

function RnnCVGPosTagger:train(tagged_sentences, learningRate, iterations)
  learningRate = learningRate or 1
  iterations = iterations or 100
  -- iterations over corps
  for itr = 1,iterations do
      print("Began the iteration: ".. itr)
    -- iterations over sentence
    print("The number of sentences:");
    print(#tagged_sentences);
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
        local initRepresent = self.lookupTable:forward(nn.LoadedLookupTable.PADDING)[1]
        -- forward the rnn
        self.rnn:forward(currentSent, initRepresent)
        -- backward the rnn
        self.rnn:backward(currentSent, initRepresent)
        -- update the parameters
        self.rnn:updateParameters(learningRate / math.sqrt(1 + iterations))
        --error('Implementing!')
    end
end
end

function RnnCVGPosTagger:validate(tagged_sentences)
    error('Not yet implemented!')
end

function RnnCVGPosTagger:tag(sentence)

    local beamSize = 3


    function extendTags(originalTags)
        local outputTags = {}
        for i = 1, #self.tagSet do
            local traili = {}
            for j = 1, #originalTags do
                table.insert(traili,  originalTags[j])
            end
            table.insert(traili,  self.tagSet[i])
            table.insert(outputTags,  traili)
        end
        return outputTags
    end
    --print(extendTags({}))
    --print(extendTags(sentence.tags))
    

    -- local misSentence = {words = sentence.words, tags = {}}
    -- local changedId = math.random(1,#sentence.words)
    -- for i = 1, # sentence.words do
    --     table.insert( misSentence.tags, sentence.tags[i])
    -- end 
    -- misSentence.tags[changedId] = self.tagSet[math.random(#self.tagSet)]

    -- print(self:scoreTagging(sentence)-self:scoreTagging(misSentence))
    local tops, trials = {{tags = {} }}, {}
    for wn = 1, #sentence.words do
        trials = {}
        for topi = 1, #tops do
            
            local newTags = extendTags(tops[topi].tags)
            for ni = 1, #newTags do
                local trySentence = {words = sentence.words, tags = newTags[ni]}
                local tryScore = self:scoreTagging(trySentence)
                table.insert(trials, {tags = newTags[ni], score = tryScore})
            end
            table.sort( trials, function (a,b) return a.score > b.score end )
        end
        
        tops = {}
        for topi = 1, beamSize do 
            table.insert(tops, trials[topi])
        end
        --print(trials)
        print(tops)
        
    end
    print(sentence.tags)
    error("Stop")

    local tagsPredName = {}
    for t = 1, #tagsPredId do
        --print(tagsPredId[t])
        tagsPredName[t] = self.tagSet[tagsPredId[t]]
    end
    return tagsPredName;
end

function RnnCVGPosTagger:scoreTagging(sentence)
    local currentSent = sentence
    local represents, indexes, tagsId = {}, {}, {}
    for wn = 1, #currentSent.tags do
        local word = currentSent['words'][wn]
        local represent = self.lookupTable:forward(word)[1]:clone()
        local index = self.lookupTable:queryIndex(word)
        local tagId = self.tagIndex[currentSent['tags'][wn]]
        if tagId == nil then
            print(currentSent['tags'][wn])
        end
        table.insert(represents,represent)
        table.insert(indexes, index)
        table.insert(tagsId, tagId)
    end
    currentSent.represents = represents
    currentSent.index = indexes
    currentSent.tagsId = tagsId
    --print(currentSent) -- @WHY output something strange
    local initRepresent = self.lookupTable:forward(nn.LoadedLookupTable.PADDING)[1]
    -- forward the rnn
    local tagsPredId, score = self.rnn:forward(currentSent, initRepresent)
    local tagsPredName = {}
    -- for t = 1, #tagsPredId do
    --     --print(tagsPredId[t])
    --     tagsPredName[t] = self.tagSet[tagsPredId[t]]
    -- end
    return score
end
