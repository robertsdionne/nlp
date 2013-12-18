require 'torch'
require 'nn'

dofile 'DataLoader.lua'
dofile 'DumbPosTagger.lua'
dofile 'RnnPosTagger.lua'
dofile 'Evaluator.lua'
dofile 'EmbeddingsUtilities.lua'
dofile 'LoadedLookupTable.lua' 

TRAIN_DATA = '../data/train_data.obj'
TRAINED_MODEL_TAGGER = '../model/trained_tagger.obj'
TRAINED_MODEL_LOOKUPTABLE = '../model/trained_lookup_table.obj'

local function main(arguments)
  -- Ported directly from POSTaggerTester.java from the assignments with a few tweaks.
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Part-of-speech tagger implemented with recursive neural networks.')
  cmd:text()
  cmd:text('Options')
  cmd:option('-iterations', 100, 'the number of training iterations')
  cmd:option('-rate_of_learning', 1, 'the learning rate') -- can't use 'learning_rate' due to a bug
  cmd:option('-reload', false, 'whether to reload trainig set')
  cmd:option('-resume', false, 'whether to resume the tranning, if yes, save the PosTagger and lookupTable')
  cmd:option('-seed', -1, 'supply a seed for the random number generator')
  cmd:option('-training_sentences', -1, 'the number of training sentences')
  cmd:option('-test', false, 'whether to evaluate on the test data')
  cmd:option('-test_sentences', -1, 'the number of test sentences')
  cmd:option('-verbose', false, 'whether to print verbosely')
  parameters = cmd:parse(arguments)
  print(parameters)
  local iterations = parameters['iterations']
  local learning_rate = parameters['rate_of_learning']
  local reload = parameters['reload']
  local resume = parameters['resume']
  local seed = parameters['seed']
  local training_sentences = parameters['training_sentences']
  local test = parameters['test']
  local test_sentences = parameters['test_sentences']
  local verbose = parameters['verbose']

  -- seed the random number generator
  if -1 ~= seed then
    torch.manualSeed(seed)
  end

  -- new the lookup table
  local lookupTable = nn.LoadedLookupTable.load()
  -- new the data loader
  local data_loader = nn.DataLoader()

  print('Loading training sentences...')
  if reload then
      print('Reloading from original corp')
      train_tagged_sentences, training_vocabulary, tags = data_loader:readTaggedSentences(TRAIN_FILENAME, training_sentences)
      -- Save Train Data
      train_data = {train_tagged_sentences, training_vocabulary, tags}
      file = torch.DiskFile(TRAIN_DATA, 'w')
      file:writeObject(train_data)
      file:close()
  else
      print('Reloading from saved obj')
      file = torch.DiskFile(TRAIN_DATA, 'r')
      train_data = file:readObject()
  end
  train_tagged_sentences, training_vocabulary, tags = train_data[1], train_data[2], train_data[3]

  print('done.')
  print('Loading in-domain dev sentences...')
  dev_in_tagged_sentences = data_loader:readTaggedSentences(DEV_IN_DOMAIN_FILENAME, test_sentences)
  print('done.')
  print('Loading out-of-domain dev sentences...')
  dev_out_tagged_sentences = data_loader:readTaggedSentences(DEV_OUT_OF_DOMAIN_FILENAME, test_sentences)
  print('Loading out-of-domain test sentences...')
  test_sentences = data_loader:readTaggedSentences(TEST_FILENAME, test_sentences)
  print('done.')

  -- init the pos tagger with lookupTable
  local pos_tagger = nn.RnnPosTagger(lookupTable, EMBEDDING_DIMENSION, EMBEDDING_DIMENSION, tags)
  -- Do the tranning or just resume the results
  if resume then
      file = torch.DiskFile(TRAINED_MODEL_TAGGER, 'r')
      pos_tagger = file:readObject()
      file = torch.DiskFile(TRAINED_MODEL_LOOKUPTABLE, 'r')
      lookupTable = file:readObject()
      pos_tagger.lookupTable = lookupTable
  else
      pos_tagger:train(train_tagged_sentences, learning_rate, iterations)
      -- Save the trained model: tagger and lookup table
      file = torch.DiskFile(TRAINED_MODEL_TAGGER, 'w')
      file:writeObject(pos_tagger)
      file:close()

      file = torch.DiskFile(TRAINED_MODEL_LOOKUPTABLE, 'w')
      file:writeObject(lookupTable)
      file:close()
  end

  --pos_tagger:validate(dev_in_tagged_sentences)

  local evaluator = nn.Evaluator()
  
  print('Evaluating on training data:')
  evaluator:evaluateTagger(pos_tagger, train_tagged_sentences, training_vocabulary, verbose)
  print('Evaluating on in-domain data:')
  evaluator:evaluateTagger(pos_tagger, dev_in_tagged_sentences, training_vocabulary, verbose)
  print('Evaluating on out-of-domain data:')
  evaluator:evaluateTagger(pos_tagger, dev_out_tagged_sentences, training_vocabulary, verbose)
  if test then
    print('Evaluating on test data:')
    evaluator:evaluateTagger(pos_tagger, test_sentences, training_vocabulary, verbose)
  end
end

main(arg)
