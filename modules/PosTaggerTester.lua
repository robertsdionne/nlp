require 'torch'
require 'nn'

dofile 'DataLoader.lua'
dofile 'Evaluator.lua'
dofile 'PosTagger.lua'

local function main(arguments)
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Part-of-speech tagger implemented with recursive neural networks.')
  cmd:text()
  cmd:text('Options')
  cmd:option('-verbose', false, 'whether to print verbosely')
  parameters = cmd:parse(arguments)
  local verbose = parameters['verbose']

  local data_loader = nn.DataLoader()

  print('Loading training sentences...')
  train_tagged_sentences, training_vocabulary = data_loader:readTaggedSentences(TRAIN_FILENAME)
  print('done.')
  print('Loading in-domain dev sentences...')
  dev_in_tagged_sentences = data_loader:readTaggedSentences(DEV_IN_DOMAIN_FILENAME)
  print('done.')
  print('Loading out-of-domain dev sentences...')
  dev_out_tagged_sentences = data_loader:readTaggedSentences(DEV_OUT_OF_DOMAIN_FILENAME)
  print('Loading out-of-domain test sentences...')
  test_sentences = data_loader:readTaggedSentences(TEST_FILENAME)
  print('done.')

  local pos_tagger = nn.PosTagger()
  -- pos_tagger:train(train_tagged_sentences)

  -- pos_tagger:validate(dev_in_tagged_sentences)

  local evaluator = nn.Evaluator()

  print('Evaluating on in-domain data:.')
  evaluator:evaluateTagger(pos_tagger, dev_in_tagged_sentences, training_vocabulary, verbose)
  print('Evaluating on out-of-domain data:.')
  evaluator:evaluateTagger(pos_tagger, dev_out_tagged_sentences, training_vocabulary, verbose)
  print('Evaluating on test data:.')
  evaluator:evaluateTagger(pos_tagger, test_sentences, training_vocabulary, verbose)
end

main(arg)
