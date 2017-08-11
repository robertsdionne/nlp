from contextlib import contextmanager

import tensorflow as tf

import dataset


class DatasetTest(tf.test.TestCase):

    def test_load_dataset(self):
        train = dataset.load_dataset(self.fake_file([
            'Hello EXCL',
            'world NN',
            '! .',
            '',
            'Hi EXCL',
            'planet NN',
            '. .',
            '',
        ]))

        self.assertEqual((
            [
                ('Hello', 'world', '!'),
                ('Hi', 'planet', '.'),
            ],
            [
                ('EXCL', 'NN', '.'),
                ('EXCL', 'NN', '.'),
            ],
        ), train)

    def test_load_datasets(self):
        train, validate, test = dataset.load_datasets()

        self.assertEqual(39815, len(train[0]))
        self.assertEqual(39815, len(train[1]))

        self.assertEqual(2716, len(validate[0]))
        self.assertEqual(2716, len(validate[1]))

        self.assertEqual(1015, len(test[0]))
        self.assertEqual(1015, len(test[1]))

    @contextmanager
    def fake_file(self, lines):
        yield lines
