-- The in-domain dev data.
DEV_IN_DOMAIN_FILENAME = '../data/en-wsj-dev.pos'
-- The out-of-domain dev data.
DEV_OUT_OF_DOMAIN_FILENAME = '../data/en-web-weblogs-dev.pos'
-- The test data.
TEST_FILENAME = '../data/en-wsj-test.tagged'
-- The training data.
TRAIN_FILENAME = '../data/en-wsj-train.pos'

local function convertSetToList(set)
  local list = {}
  local i = 1
  for item, _ in pairs(set) do
    list[i] = item
    i = i + 1
  end
  table.sort(list)
  return list
end

-- Function to load a data file into a table of nn.TaggedSentences.
function loadData(data_filename)
  -- Open the data file.
  local data_file = torch.DiskFile(data_filename)
  local tagged_sentences, sentence_words, sentence_tags, word_set, tag_set = {}, {}, {}, {}, {}
  local i, j = 1, 1
  repeat
    -- Use pcall to safely invoke data_file:readString('*l') since readString throws an error upon
    -- the end-of-file marker.
    -- The argument '*l' indicates to read one line of the file at a time.
    -- If we reach the end-of-file marker, success will be false.
    local success, line = pcall(data_file.readString, data_file, '*l')

    if '' ~= line then
      -- If the line is not empty, read its (word, tag) pair.
      for word, tag in line:gmatch('(.*)%s+(.*)') do
        sentence_words[j] = word
        sentence_tags[j] = tag
        word_set[word] = true
        tag_set[tag] = true
        j = j + 1
      end
    else
      -- If the line is empty, assemble the sentence_words and sentence_tags
      -- into a nn.TaggedSentence.
      tagged_sentences[i] = nn.TaggedSentence(sentence_words, sentence_tags)
      sentence_words = {}
      sentence_tags = {}
      j = 1
      i = i + 1
    end
  until not success

  -- Close the data file.
  data_file:close()

  -- Return all the nn.TaggedSentences.
  return tagged_sentences, convertSetToList(word_set), convertSetToList(tag_set)
end
