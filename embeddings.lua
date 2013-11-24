require 'torch'

local EMBEDDING_COUNT = 130000
local EMBEDDING_DIMENSION = 50

function loadEmbeddings(embeddings_filename)
  local embeddings_file = torch.DiskFile(embeddings_filename)
  local raw_embeddings = torch.Tensor(
    embeddings_file:readDouble(EMBEDDING_DIMENSION * EMBEDDING_COUNT))
  local embeddings = torch.Tensor(EMBEDDING_COUNT, EMBEDDING_DIMENSION)
  embeddings:copy(raw_embeddings)
  return embeddings:t()
end
