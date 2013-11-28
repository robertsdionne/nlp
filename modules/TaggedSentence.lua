local TaggedSentence = torch.class('nn.TaggedSentence')

function TaggedSentence:__init(words, tags)
  assert(#words == #tags, 'Word count must match tag count!')
  self.words = words
  self.tags = tags
end

function TaggedSentence:size()
  return #self.words
end

function TaggedSentence:toString()
  local size = self:size()
  local result = ''
  for i = 1, size - 1 do
    result = result .. self.words[i] .. '_' .. self.tags[i] .. ' '
  end
  return result .. self.words[size] .. '_' .. self.tags[size]
end
