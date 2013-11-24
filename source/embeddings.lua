require 'torch'

EMBEDDING_COUNT = 130000
EMBEDDING_DIMENSION = 50

function loadEmbeddings(embeddings_filename)
  embeddings_file = torch.DiskFile(embeddings_filename)
  raw_embeddings = torch.Tensor(
    embeddings_file:readDouble(EMBEDDING_DIMENSION * EMBEDDING_COUNT))
  embeddings = torch.Tensor(EMBEDDING_COUNT, EMBEDDING_DIMENSION)
  embeddings:copy(raw_embeddings)
  return embeddings:t()
end
