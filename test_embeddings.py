from contextlib import contextmanager

import numpy as np
import tensorflow as tf

import embeddings


class EmbeddingsTest(tf.test.TestCase):

    def test_load(self):
        words, indices, vectors = embeddings.load(
            self.fake_file([
                'hello',
                'world',
                '!',
            ]),
            self.fake_file([
                '0.1 0.2 0.3',
                '0.4 0.5 0.6',
                '0.7 0.8 0.9',
            ])
        )

        self.assertEqual(['hello', 'world', '!'], words)
        self.assertEqual({'hello': 0, 'world': 1, '!': 2}, indices)
        self.assertAllEqual(np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ], dtype=np.float32), vectors)

    def test_load_embeddings(self):
        words, indices, vectors = embeddings.load(
            open(tf.flags.FLAGS.embeddings_index), open(tf.flags.FLAGS.embeddings_vectors))

        self.assertEqual(130000, len(words))
        self.assertEqual(130000, len(indices))
        self.assertEqual((130000, 50), vectors.shape)

    @contextmanager
    def fake_file(self, lines):
        yield lines
