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
  cmd:option('-test', false, 'whether to evaluate on the test data')
  cmd:option('-verbose', false, 'whether to print verbosely')
  cmd:option('-reload', false, 'whether to reload trainnig set')
  cmd:option('-resume', false, 'whether to resume the tranning, if yes, save the PosTagger and lookupTable')
  parameters = cmd:parse(arguments)
  print(parameters)
  local verbose, test = parameters['verbose'], parameters['test']
  local reload = parameters['reload']
  local resume = parameters['resume']
  -- new the lookup table
  local lookupTable = nn.LoadedLookupTable.load()
  -- new the data loader
  local data_loader = nn.DataLoader()

  print('Loading training sentences...')
  if reload then
      print('Reloading from original corp')
      train_tagged_sentences, training_vocabulary, tags = data_loader:readTaggedSentences(TRAIN_FILENAME)
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
  dev_in_tagged_sentences = data_loader:readTaggedSentences(DEV_IN_DOMAIN_FILENAME)
  print('done.')
  print('Loading out-of-domain dev sentences...')
  dev_out_tagged_sentences = data_loader:readTaggedSentences(DEV_OUT_OF_DOMAIN_FILENAME)
  print('Loading out-of-domain test sentences...')
  test_sentences = data_loader:readTaggedSentences(TEST_FILENAME)
  print('done.')

  -- init the pos tagger with lookupTable
  local pos_tagger = nn.RnnPosTagger(lookupTable, EMBEDDING_DIMENSION, 50, tags)
<<<<<<< HEAD
  pos_tagger:train(train_tagged_sentences)
  --pos_tagger:validate(dev_in_tagged_sentences)
=======
  -- Do the tranning or just resume the results
  if resume then
      local _ = 1;
  else
      pos_tagger:train(train_tagged_sentences)
      -- Save the trained model: tagger and lookup table
      file = torch.DiskFile(TRAINED_MODEL_TAGGER, 'w')
      file:writeObject(pos_tagger)
      file:close()

      file = torch.DiskFile(TRAINED_MODEL_LOOKUPTABLE, 'w')
      file:writeObject(lookupTable)
      file:close()
  end

  pos_tagger:validate(dev_in_tagged_sentences)
>>>>>>> 35960e989ebbf555b28fa4edf50f6a29a7e1f5f8

  local evaluator = nn.Evaluator()

  print('Evaluating on training data:')
  --evaluator:evaluateTagger(pos_tagger, train_tagged_sentences, training_vocabulary, verbose)
  print('Evaluating on in-domain data:')
  --evaluator:evaluateTagger(pos_tagger, dev_in_tagged_sentences, training_vocabulary, verbose)
  print('Evaluating on out-of-domain data:')
  evaluator:evaluateTagger(pos_tagger, dev_out_tagged_sentences, training_vocabulary, verbose)
  if test then
    print('Evaluating on test data:')
    evaluator:evaluateTagger(pos_tagger, test_sentences, training_vocabulary, verbose)
  end
end

main(arg)
