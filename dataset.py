import tensorflow as tf


tf.flags.DEFINE_string('dataset-train', 'data/en-wsj-train.pos', 'The training data.')
tf.flags.DEFINE_string('dataset-validate-0', 'data/en-wsj-dev.pos', 'The validation data 0.')
tf.flags.DEFINE_string('dataset-validate-1', 'data/en-web-weblogs-dev.pos', 'The validation data 1.')
tf.flags.DEFINE_string('dataset-test', 'data/en-web-test.tagged', 'The test data.')
FLAGS = tf.flags.FLAGS


def load_datasets(train_file=None, validate_file_0=None, validate_file_1=None, test_file=None):
    if train_file is None:
        train_file = open(FLAGS.dataset_train)

    if validate_file_0 is None:
        validate_file_0 = open(FLAGS.dataset_validate_0)

    if validate_file_1 is None:
        validate_file_1 = open(FLAGS.dataset_validate_1)

    if test_file is None:
        test_file = open(FLAGS.dataset_test)

    train, validate, test = (
        load_dataset(train_file), load_dataset(validate_file_0, validate_file_1), load_dataset(test_file))
    return train, validate, test


def load_dataset(*dataset_files):
    examples = []
    labels = []
    for dataset_file in dataset_files:
        with dataset_file as file:
            pairs = []
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    example, label = zip(*pairs)
                    pairs = []
                    examples.append(example)
                    labels.append(label)
                else:
                    pairs.append(line.split())
    return examples, labels



class Dataset(object):

    def __init__(self, examples, labels):
        self.examples = examples
        self.labels = labels
