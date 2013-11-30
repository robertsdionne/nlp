require 'torch'
require 'nn'

dofile 'TaggedSentence.lua'
dofile 'DataLoader.lua'
dofile 'PosTagger.lua'

local function main(arguments)
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Part-of-speech tagger implemented with recursive neural networks.')
  cmd:text()
  cmd:text('Options')
  parameters = cmd:parse(arguments)
  
  print('Loading training sentences...')
  trainTaggedSentences, trainingVocabulary = readTaggedSentences(TRAIN_FILENAME)
  print('done.')
  print('Loading in-domain dev sentences...')
  devInTaggedSentences = readTaggedSentences(DEV_IN_DOMAIN_FILENAME)
  print('done.')
  print('Loading out-of-domain dev sentences...')
  devOutTaggedSentences = readTaggedSentences(DEV_OUT_OF_DOMAIN_FILENAME)
  print('Loading out-of-domain test sentences...')
  testSentences = readTaggedSentences(TEST_FILENAME)
  print('done.')

  posTagger = nn.PosTagger()
  -- posTagger.train(trainTaggedSentences)

  -- posTagger.validate(devInTaggedSentences)

  print('Evaluating on in-domain data:.')
  -- evaluateTagger(posTagger, devInTaggedSentences, trainingVocabulary)
  print('Evaluating on out-of-domain data:.')
  -- evaluateTagger(posTagger, devOutTaggedSentences, trainingVocabulary)
  print('Evaluating on test data:.')
  -- evaluateTagger(posTagger, testSentences, trainingVocabulary)
end

main(arg)
