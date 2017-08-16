import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf

import dataset
import embeddings


DECAY = 0.999
EPSILON = 1e-8
SIZE = 3
CHANNELS = 128


def main():
    words, indices, vectors = embeddings.load()
    train, validate, test, parts_of_speech = dataset.load_datasets(indices={
        **{word: indices[word] + 2 for word in indices},
        **{
            '<padding/>': 0,
            '<unknown/>': 1,
        },
    })

    with tf.Session() as session:
        sentence_batch = tf.placeholder(tf.int64, shape=[None, None], name='sentence_batch')
        labels_batch = tf.placeholder(tf.int64, shape=[None, None], name='labels_batch')
        sentence_length = tf.placeholder(tf.int32, shape=[None], name='sentence_length')
        print(sentence_batch)
        print(labels_batch)
        print(sentence_length)

        with tf.variable_scope('embeddings'):
            word_embeddings = tf.concat([
                tf.zeros([1, vectors.shape[1]], name='padding'),
                tf.get_variable('unknown', shape=[1, vectors.shape[1]]),
                vectors,
            ], 0, name='word_embeddings')
            print(word_embeddings)

            word_batch = tf.nn.embedding_lookup(word_embeddings, sentence_batch)
            word_batch = tf.expand_dims(word_batch, axis=1, name='word_batch')
            print(word_batch)

            normalized_word_batch = tf.nn.relu(batch_normalization(word_batch), name='normalized_word_batch')
            print(normalized_word_batch)

        with tf.variable_scope('bytenet'):
            bias_initializer = tf.zeros_initializer()

            w0 = tf.get_variable('w0', shape=[1, SIZE, word_batch.get_shape()[-1], CHANNELS])
            b0 = tf.get_variable('b0', shape=[CHANNELS], initializer=bias_initializer)

            h0 = tf.identity(tf.nn.atrous_conv2d(normalized_word_batch, w0, 2**0, 'SAME') + b0, name='h0')
            normalized_h0 = tf.nn.relu(batch_normalization(h0))

            w1 = tf.get_variable('w1', shape=[1, SIZE, CHANNELS, CHANNELS])
            b1 = tf.get_variable('b1', shape=[CHANNELS], initializer=bias_initializer)

            h1 = tf.identity(tf.nn.atrous_conv2d(normalized_h0, w1, 2**1, 'SAME') + b1, name='h1')
            normalized_h1 = tf.nn.relu(batch_normalization(h1))

            w2 = tf.get_variable('w2', shape=[1, SIZE, CHANNELS, CHANNELS])
            b2 = tf.get_variable('b2', shape=[CHANNELS], initializer=bias_initializer)

            h2 = tf.identity(tf.nn.atrous_conv2d(normalized_h1, w2, 2**2, 'SAME') + b2 + normalized_h0, name='h2')
            normalized_h2 = tf.nn.relu(batch_normalization(h2))

            w3 = tf.get_variable('w3', shape=[1, SIZE, CHANNELS, CHANNELS])
            b3 = tf.get_variable('b3', shape=[CHANNELS], initializer=bias_initializer)

            h3 = tf.identity(tf.nn.atrous_conv2d(normalized_h2, w3, 2**3, 'SAME') + b3, name='h3')
            normalized_h3 = tf.nn.relu(batch_normalization(h3))

            w4 = tf.get_variable('w4', shape=[1, SIZE, CHANNELS, CHANNELS])
            b4 = tf.get_variable('b4', shape=[CHANNELS], initializer=bias_initializer)

            h4 = tf.identity(tf.nn.atrous_conv2d(normalized_h3, w4, 2**4, 'SAME') + b4 + normalized_h2, name='h4')
            normalized_h4 = tf.nn.relu(batch_normalization(h4))

            w5 = tf.get_variable('w5', shape=[1, SIZE, CHANNELS, CHANNELS])
            b5 = tf.get_variable('b5', shape=[CHANNELS], initializer=bias_initializer)

            h5 = tf.identity(tf.nn.atrous_conv2d(normalized_h4, w5, 2**5, 'SAME') + b5, name='h5')
            normalized_h5 = tf.nn.relu(batch_normalization(h5))

            w6 = tf.get_variable('w6', shape=[1, SIZE, CHANNELS, CHANNELS])
            b6 = tf.get_variable('b6', shape=[CHANNELS], initializer=bias_initializer)

            h6 = tf.identity(tf.nn.atrous_conv2d(normalized_h5, w6, 2**0, 'SAME') + b6 + normalized_h4, name='h6')
            normalized_h6 = tf.nn.relu(batch_normalization(h6))

            w7 = tf.get_variable('w7', shape=[1, SIZE, CHANNELS, CHANNELS])
            b7 = tf.get_variable('b7', shape=[CHANNELS], initializer=bias_initializer)

            h7 = tf.identity(tf.nn.atrous_conv2d(normalized_h6, w7, 2**1, 'SAME') + b7, name='h7')
            normalized_h7 = tf.nn.relu(batch_normalization(h7))

            w8 = tf.get_variable('w8', shape=[1, SIZE, CHANNELS, CHANNELS])
            b8 = tf.get_variable('b8', shape=[CHANNELS], initializer=bias_initializer)

            h8 = tf.identity(tf.nn.atrous_conv2d(normalized_h7, w8, 2**2, 'SAME') + b8 + normalized_h6, name='h8')
            normalized_h8 = tf.nn.relu(batch_normalization(h8))

            w9 = tf.get_variable('w9', shape=[1, SIZE, CHANNELS, CHANNELS])
            b9 = tf.get_variable('b9', shape=[CHANNELS], initializer=bias_initializer)

            h9 = tf.identity(tf.nn.atrous_conv2d(normalized_h8, w9, 2**3, 'SAME') + b9, name='h9')
            normalized_h9 = tf.nn.relu(batch_normalization(h9))

            w10 = tf.get_variable('w10', shape=[1, SIZE, CHANNELS, CHANNELS])
            b10 = tf.get_variable('b10', shape=[CHANNELS], initializer=bias_initializer)

            h10 = tf.identity(tf.nn.atrous_conv2d(normalized_h9, w10, 2**4, 'SAME') + b10 + normalized_h8, name='h10')
            normalized_h10 = tf.nn.relu(batch_normalization(h10))

            w11 = tf.get_variable('w11', shape=[1, SIZE, CHANNELS, CHANNELS])
            b11 = tf.get_variable('b11', shape=[CHANNELS], initializer=bias_initializer)

            h11 = tf.identity(tf.nn.atrous_conv2d(normalized_h10, w11, 2**5, 'SAME') + b11, name='h11')
            normalized_h11 = tf.nn.relu(batch_normalization(h11))

            w12 = tf.get_variable('w12', shape=[1, SIZE, CHANNELS, len(parts_of_speech)])
            b12 = tf.get_variable('b12', shape=[len(parts_of_speech)], initializer=bias_initializer)

            h12 = tf.identity(tf.nn.atrous_conv2d(normalized_h11, w12, 2**0, 'SAME') + b12, name='h12')
            h12 = tf.squeeze(h12, axis=[1])

        weights = tf.sequence_mask(sentence_length, name='weights')
        correct = tf.logical_and(tf.equal(tf.argmax(h12, -1), labels_batch), weights)
        weights = tf.cast(weights, tf.float32)
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(weights)
        sequence_loss = tf.contrib.seq2seq.sequence_loss(h12, labels_batch, weights)
        y = tf.nn.softmax(h12, name='y')

        train_step = tf.train.AdamOptimizer(1e-3).minimize(sequence_loss)

        session.run(tf.global_variables_initializer())

        M = 128

        for e in range(20):
            examples, labels = train
            indices = np.arange(len(examples))
            np.random.shuffle(indices)

            for i in range(len(examples) // M):
                index = indices[i:i+M]
                sentences = np.take(examples, index)
                lengths = list(map(len, sentences))
                sentences = pad_sequences(sentences, dtype=np.int64, padding='post')
                label = np.take(labels, index)
                label = pad_sequences(label, dtype=np.int64, padding='post')

                train_loss, train_accuracy, _ = session.run([sequence_loss, accuracy, train_step], feed_dict={
                    sentence_batch: sentences,
                    labels_batch: label,
                    sentence_length: lengths,
                })

                print(e, i, train_loss, train_accuracy)



def batch_normalization(x):
    with tf.variable_scope('batch_norm'):
        population_mean = tf.Variable(tf.zeros([1, x.get_shape()[-1]]), trainable=False, name='population_mean')
        population_variance = tf.Variable(tf.zeros([1, x.get_shape()[-1]]), trainable=False, name='population_variance')

        mu, sigma2 = tf.nn.moments(x, axes=[0, 2])
        train_mean = tf.assign(population_mean, DECAY * population_mean + (1 - DECAY) * mu)
        train_variance = tf.assign(population_variance, DECAY * population_variance + (1 - DECAY) * sigma2)

        with tf.control_dependencies([train_mean, train_variance]):
            normalized_x = tf.nn.batch_normalization(x, mu, sigma2, None, None, EPSILON)

    return normalized_x


if __name__ == '__main__':
    main()
