#!/usr/bin/env python3
import tensorflow as tf

from monty import data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of data items to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 10,
                            """Number of learning epochs.""")
tf.app.flags.DEFINE_float('learning_rate', 0.0001, """Learning rate.""")
tf.app.flags.DEFINE_integer('num_features', 2700,
                            """Number of features (types of RNA transcripts) in the data.""")
tf.app.flags.DEFINE_string('dataset_path', 'resources/PBMC.csv',
                           """Path to the dataset.""")
tf.app.flags.DEFINE_integer('minimum_library_size', 50,
                            """Minimum sum of expressed genes that a measurement needs to have,
                            in order to be processed""")
tf.app.flags.DEFINE_integer('minimum_expressed_genes', 20,
                            """Minimum count of nonzero gene expression that a measurement needs to have,
                            in order to be processed""")
tf.app.flags.DEFINE_integer('impute_iterations', 20,
                            """How many times the imputation should iterate at each step""")


def encoder(x):
    x = tf.layers.dense(x, 100)
    return x


def decoder(x):
    x = tf.layers.dense(x, 2700, activation=lambda logits: tf.nn.sigmoid(logits) * 2)
    return x


if __name__ == '__main__':
    dataset_file_path = data.download_if_not_present(FLAGS.dataset_path)
    dataset = data.create_dataset(dataset_file_path,
                                  num_features=FLAGS.num_features,
                                  num_epochs=FLAGS.num_epochs,
                                  shuffle=True,
                                  shuffle_buffer_size=1000)
    iterator = dataset.batch(batch_size=FLAGS.batch_size, drop_remainder=True).make_one_shot_iterator()

    input = iterator.get_next()

    ae_out = decoder(encoder(input))
    loss = tf.reduce_mean(tf.square(ae_out - input))
    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

    sess = tf.Session()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        while True:
            _, loss_value = sess.run([train_op, loss])
            print(loss_value)
