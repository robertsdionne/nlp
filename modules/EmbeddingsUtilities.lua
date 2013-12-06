-- The path to the Collobert and Weston 2011 word embeddings file.
EMBEDDINGS_FILENAME = '../embeddings/embeddings.txt'
-- The path to the Collobert and Weston 2011 words file.
WORDS_FILENAME = '../embeddings/words.lst'

-- The number of word embeddings.
local EMBEDDING_COUNT = 130000
-- The vector dimension of the embeddings.
EMBEDDING_DIMENSION = 50

-- Returns a torch.Tensor containing as rows each word embedding loaded from the
-- given files.
function loadEmbeddings(embeddings_filename, words_filename)
  local embeddings_file = torch.DiskFile(embeddings_filename)
  local raw_embeddings = torch.Tensor(
    embeddings_file:readDouble(EMBEDDING_DIMENSION * EMBEDDING_COUNT))
  embeddings_file:close()
  local embeddings = torch.Tensor(EMBEDDING_COUNT, EMBEDDING_DIMENSION)
  embeddings:copy(raw_embeddings)
  words_file = torch.DiskFile(words_filename)
  local word_to_index = {}
  local index_to_word = {}
  for index=1,EMBEDDING_COUNT do
    word = words_file:readString('*l')
    word_to_index[word] = index
    index_to_word[index] = word
  end
  words_file:close()
  return embeddings, word_to_index, index_to_word
end
