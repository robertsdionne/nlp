require 'torch'

loadfile('embeddings.lua')()

-- Read the Collobert and Weston 2011 word embeddings.
embeddings = loadEmbeddings('embeddings/embeddings.txt')

-- Print the dimensions.
print(embeddings:size())

-- Print the embedding for word 12 by first taking the transpose.
print(embeddings:t()[12])
