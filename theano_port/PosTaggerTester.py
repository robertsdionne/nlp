#!/usr/bin/env python
import argparse
from DataLoader import DataLoader
from DumbPosTagger import DumbPosTagger
from Evaluator import Evaluator

def main():
  cmd = argparse.ArgumentParser(
      description = 'Part-of-speech tagger implemented with recursive neural networks.')
  cmd.add_argument('--iterations', dest = 'iterations', type = int, default = 1000,
      help = 'the number of training iterations')
  cmd.add_argument('--learning_rate', dest = 'learning_rate', type = float, default = 0.1,
      help = 'the learning rate')
  cmd.add_argument('--reload', action = 'store_const', dest = 'reload', const = True,
      default = False, help = 'whether to reload the training set')
  cmd.add_argument('--resume', action = 'store_const', dest = 'resume', const = True,
      default = False, help = 'whether to resume training')
  cmd.add_argument('--training_sentences', dest = 'training_sentences', type = int,
    help = 'the number of training sentences')
  cmd.add_argument('--test', action = 'store_const', dest = 'test', const = True,
      default = False, help = 'whether to evaluate on the test data')
  cmd.add_argument('--test_sentences', dest = 'test_sentences', type = int,
    help = 'the number of test sentences')
  cmd.add_argument('--verbose', action = 'store_const', dest = 'verbose', const = True,
      default = False, help = 'whether to print verbosely')
  arguments = cmd.parse_args()
  print(arguments)
  # lookup_table = LoadedLookupTable.load()
  data_loader = DataLoader()

  print('Loading training sentences...')
  train_tagged_sentences, training_vocabulary, tags, _, _ = data_loader.read_tagged_sentences(
      DataLoader.TRAIN_FILENAME, arguments.training_sentences)
  print('done.')
  print('Loading in-domain dev sentences...')
  dev_in_tagged_sentences, _, _, _, _ = data_loader.read_tagged_sentences(
      DataLoader.DEV_IN_DOMAIN_FILENAME, arguments.test_sentences)
  print('done.')
  print('Loading out-of-domain dev sentences...')
  dev_out_tagged_sentences, _, _, _, _ = data_loader.read_tagged_sentences(
      DataLoader.DEV_OUT_OF_DOMAIN_FILENAME, arguments.test_sentences)
  print('done.')
  print('Loading out-of-domain test sentences...')
  test_sentences, _, _, _, _ = data_loader.read_tagged_sentences(
      DataLoader.TEST_FILENAME, arguments.test_sentences)
  print('done.')

  pos_tagger = DumbPosTagger()
  pos_tagger.train(train_tagged_sentences, arguments.learning_rate, arguments.iterations)

  evaluator = Evaluator()
  print('Evaluating on training data:')
  evaluator.evaluate_tagger(
      pos_tagger, train_tagged_sentences, training_vocabulary, arguments.verbose)
  print('Evaluating on in-domain data:')
  evaluator.evaluate_tagger(
      pos_tagger, dev_in_tagged_sentences, training_vocabulary, arguments.verbose)
  print('Evaluating on out-of-domain data:')
  evaluator.evaluate_tagger(
      pos_tagger, dev_out_tagged_sentences, training_vocabulary, arguments.verbose)
  if arguments.test:
    print('Evaluating on test data:')
    evaluator.evaluate_tagger(
        pos_tagger, test_sentences, training_vocabulary, arguments.verbose)

if '__main__' == __name__:
  main()
