import numpy as np
import tensorflow as tf


tf.flags.DEFINE_string('embeddings-index', 'embeddings/words.lst', 'The path to the embeddings index file.')
tf.flags.DEFINE_string('embeddings-vectors', 'embeddings/embeddings.txt', 'The path to the embeddings file.')
FLAGS = tf.flags.FLAGS


def load(index_file=None, vectors_file=None):
    if index_file is None:
        index_file = open(FLAGS.embeddings_index)

    if vectors_file is None:
        vectors_file = open(FLAGS.embeddings_vectors)

    with index_file as file:
        words = [word.strip() for word in file]

    indices = {word: index for index, word in enumerate(words)}

    with vectors_file as file:
        vectors = np.array([np.fromiter(map(float, line.split()), dtype=np.float32) for line in file])

    return words, indices, vectors
