local LoadedLookupTable, parent = torch.class(
    'nn.LoadedLookupTable', 'nn.LookupTable')

dofile 'EmbeddingsUtils.lua'

local function prepareInput(self, input)
  if type(input) == 'number' then
    --Convert an index to a tensor.
    tensor = torch.Tensor(1)
    tensor[1] = input
    return tensor
  elseif type(input) == 'string' then
    --Convert a word to an index to a tensor.
    tensor = torch.Tensor(1)
    tensor[1] = self:queryIndex(word)
    return tensor
  else
    --Assume we have a tensor already.
    return input
  end
end

function LoadedLookupTable.load(embeddings_filename, words_filename)
  return nn.LoadedLookupTable(embeddings_filename, words_filename,
      loadEmbeddings(embeddings_filename, words_filename))
end

function LoadedLookupTable:__init(embeddings_filename, words_filename,
    embeddings, word_to_index, index_to_word)
  self.embeddings_filename = embeddings_filename
  self.words_filename = words_filename
  parent.__init(self, embeddings:size(1), embeddings:size(2))
  self:reset(embeddings)
  self.word_to_index = word_to_index
  self.index_to_word = index_to_word
end

function LoadedLookupTable:backward(input, gradOutput, scale)
  return parent.backward(self, prepareInput(self, input), gradOutput, scale)
end

function LoadedLookupTable:backwardUpdate(input, gradOutput, lr)
  return parent.backwardUpdate(self, prepareInput(self, input), gradOutput, lr)
end

function LoadedLookupTable:forward(input)
  return parent.forward(self, prepareInput(self, input))
end

function LoadedLookupTable:queryIndex(word)
  return self.word_to_index[word]
end

function LoadedLookupTable:queryWord(index)
  return self.index_to_word[index]
end

function LoadedLookupTable:reset(embeddings)
  self.weight = embeddings
end
