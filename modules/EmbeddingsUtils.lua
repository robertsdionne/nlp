local EMBEDDING_COUNT = 130000
local EMBEDDING_DIMENSION = 50

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
