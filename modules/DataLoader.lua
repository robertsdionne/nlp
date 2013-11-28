DEV_IN_DOMAIN_FILENAME = '../data/en-wsj-dev.pos'
DEV_OUT_OF_DOMAIN_FILENAME = '../data/en-web-weblogs-dev.pos'
TEST_FILENAME = '../data/en-wsj-test.tagged'
TRAIN_FILENAME = '../data/en-wsj-train.pos'

function loadData(data_filename)
  local data_file = torch.DiskFile(data_filename)
  local tagged_sentences, words, tags = {}, {}, {}
  local i, j = 1, 1
  repeat
    local success, line = pcall(data_file.readString, data_file, '*l')
    if '' ~= line then
      for word, tag in line:gmatch('(.*)%s+(.*)') do
        words[j] = word
        tags[j] = tag
        j = j + 1
      end
    else
      tagged_sentences[i] = nn.TaggedSentence(words, tags)
      words = {}
      tags = {}
      j = 1
      i = i + 1
    end
  until not success
  data_file:close()
  return tagged_sentences
end
