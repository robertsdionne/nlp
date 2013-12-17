import argparse
from DataLoader import DataLoader

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
  train_tagged_sentences, training_vocabulary, tags = data_loader.read_tagged_sentences(
      DataLoader.TRAIN_FILENAME, arguments.training_sentences)
  print('done.')
  print('Loading in-domain dev sentences...')
  dev_in_tagged_sentences = data_loader.read_tagged_sentences(
      DataLoader.DEV_IN_DOMAIN_FILENAME, arguments.test_sentences)
  print('done.')
  print('Loading out-of-domain dev sentences...')
  dev_out_tagged_sentences = data_loader.read_tagged_sentences(
      DataLoader.DEV_OUT_DOMAIN_FILENAME, arguments.test_sentences)
  print('done.')
  print('Loading out-of-domain test sentences...')
  test_sentences = data_loader.read_tagged_sentences(
      DataLoader.TEST_FILENAME, arguments.test_sentences)
  print('done.')

if __name__ == '__main__':
  main()
