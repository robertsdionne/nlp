require 'torch'
require 'nn'

dofile 'EmbeddingsUtilities.lua'

local LoadedLookupTable, parent = torch.class(
    'nn.LoadedLookupTable', 'nn.LookupTable')

-- Converts a word or word index to a torch.Tensor containing the index.
local function prepareInput(self, input)
  if type(input) == 'number' then
    --Convert an index to a tensor.
    tensor = torch.Tensor(1)
    tensor[1] = input
    return tensor
  elseif type(input) == 'string' then
    --Convert a word to an index to a tensor.
    tensor = torch.Tensor(1)
    tensor[1] = self:queryIndex(input)
    return tensor
  else
    --Assume we have a tensor already.
    return input
  end
end

LoadedLookupTable.PADDING = 1739
LoadedLookupTable.UNKNOWN = 1740

-- Load the Collobert and Weston 2011 word embeddings into a
-- new nn.LoadedLookupTable.
function LoadedLookupTable.load()
  return nn.LoadedLookupTable(
      loadEmbeddings(EMBEDDINGS_FILENAME, WORDS_FILENAME))
end

-- Construct a nn.LoadedLookupTable.
function LoadedLookupTable:__init(embeddings, word_to_index, index_to_word)
  parent.__init(self, embeddings:size(1), embeddings:size(2))
  self:reset(embeddings)
  self.word_to_index = word_to_index
  self.index_to_word = index_to_word
end

-- Backwards propagation.
function LoadedLookupTable:backward(input, gradOutput, scale)
  return parent.backward(self, prepareInput(self, input), gradOutput, scale)
end

-- Backwards propagation and update.
function LoadedLookupTable:backwardUpdate(input, gradOutput, lr)
  return parent.backwardUpdate(self, prepareInput(self, input), gradOutput, lr)
end

-- Forward propagation.
function LoadedLookupTable:forward(input)
  return parent.forward(self, prepareInput(self, input))
end

-- Returns the index of the given word.
function LoadedLookupTable:queryIndex(word)
  local index = self.word_to_index[word]
  if not index then
    index = self.word_to_index['UNKNOWN']
  end
  return index
end

-- Returns the word at the given index.
function LoadedLookupTable:queryWord(index)
  return self.index_to_word[index]
end

-- Resets the lookup table values to the given embeddings torch.Tensor.
function LoadedLookupTable:reset(embeddings)
  self.weight = embeddings
end
