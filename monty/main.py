#!/usr/bin/env python3
import tensorflow as tf

from monty import data, FLAGS


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
    dataset = data.drop_outliers(dataset, FLAGS.minimum_expressed_genes, FLAGS.minimum_library_size)
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
